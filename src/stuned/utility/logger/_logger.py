import os
import torch
import traceback
import subprocess
import re
from stnd.utility.logger import make_logger, extract_by_regex_from_url


# local modules
from ..utils import (
    kill_processes,
    read_json,
    retrier_factory,
    is_number,
    apply_func_to_dict_by_nested_key,
    get_leaves_of_nested_dict,
    pretty_json,
    is_nested_dict,
    compute_tensor_cumsums,
)


# Tensorboard
TB_CREDENTIALS_FIELDS = [
    "refresh_token",
    "token_uri",
    "client_id",
    "client_secret",
    "scopes",
    "type",
]
TB_CREDENTIALS_DEFAULT = {
    "scopes": ["openid", "https://www.googleapis.com/auth/userinfo.email"],
    "type": "authorized_user",
}
TB_URL_COLUMN = "Tensorboard url"
TB_LOG_FOLDER = "tb_log"
TB_TIME_TO_LOG_LAST_EPOCH = 5
TB_OUTPUT_BEFORE_LINK = "View your TensorBoard at: "
TB_OUTPUT_AFTER_LINK = "\n"
TB_OUTPUT_SIZE_LIMIT = 10 * 1024
READ_BUFSIZE = 1024
TENSORBOARD_FINISHED = "Total uploaded:"
MAX_TIME_BETWEEN_TB_LOG_UPDATES = 60
MAX_TIME_TO_WAIT_FOR_TB_TO_SAVE_DATA = 1800


URL_KEY_RE = re.compile(r"key=([^&#]+)")
URL_SPREADSHEET_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")


def try_to_log_in_tb(
    logger,
    dict_to_log,
    step,
    step_offset=0,
    flush=False,
    text=False,
    same_plot=False,
):
    if logger.tb_run:
        if text:
            for key, value in dict_to_log.items():
                if isinstance(value, dict):
                    value = pretty_json(value)
                assert isinstance(value, str)
                logger.tb_run.add_text(key, value, global_step=step)
        else:
            tb_log(
                logger.tb_run,
                dict_to_log,
                step,
                step_offset=step_offset,
                flush=flush,
                same_plot=same_plot,
            )


def assert_tb_credentials(credentials_path):
    def assert_field(field_name, credentials_dict):
        assert field_name in credentials_dict
        if field_name in TB_CREDENTIALS_DEFAULT:
            assert (
                credentials_dict[field_name]
                == TB_CREDENTIALS_DEFAULT[field_name]
            )
        else:
            assert isinstance(credentials_dict[field_name], str)
            assert len(credentials_dict[field_name])

    assert os.path.exists(credentials_path)

    credentials = read_json(credentials_path)
    for field in TB_CREDENTIALS_FIELDS:
        assert_field(field, credentials)


def run_tb_folder_listener(log_folder, exp_name, description=None):
    cmd_as_list = [
        "tensorboard",
        "dev",
        "upload",
        "--logdir",
        os.path.join(log_folder, TB_LOG_FOLDER),
        "--name",
        exp_name,
    ]

    if description is not None:
        cmd_as_list += ["--description", description]

    proc = subprocess.Popen(
        cmd_as_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return proc


def get_tb_url(logger, tb_process_spawner):
    def read_proc_output(stream, total_output_size):
        size_read_so_far = 0
        result = ""

        while (
            stream
            and size_read_so_far < total_output_size
            and not TENSORBOARD_FINISHED in result
        ):
            result += stream.read(READ_BUFSIZE).decode("utf-8")
            size_read_so_far += READ_BUFSIZE

        return result

    def extract_link(output, logger):
        def assert_in_output(expected_string, output):
            assert (
                expected_string in output
            ), 'Expected "{}" in output:\n{}'.format(expected_string, output)

        assert_in_output(TB_OUTPUT_BEFORE_LINK, output)
        assert_in_output(TB_OUTPUT_AFTER_LINK, output)
        result = output.split(TB_OUTPUT_BEFORE_LINK)[1]
        return result.split(TB_OUTPUT_AFTER_LINK)[0]

    def final_func(logger):
        logger.error(
            "Could not get a tb link.\nReason: {}".format(
                traceback.format_exc()
            )
        )
        return None

    @retrier_factory(logger, final_func)
    def try_to_extract_link(tb_process_spawner):
        tb_proc = tb_process_spawner()

        out = read_proc_output(tb_proc.stdout, TB_OUTPUT_SIZE_LIMIT)

        if tb_proc.poll() is None:
            # process is alive
            kill_processes([tb_proc.pid], logger)
        else:
            err = read_proc_output(tb_proc.stderr, TB_OUTPUT_SIZE_LIMIT)
            if err:
                logger.error(err)

        assert TENSORBOARD_FINISHED in out

        tb_url = extract_link(out, logger)

        return tb_url

    tb_url = try_to_extract_link(tb_process_spawner)

    return tb_url


def tb_log(
    tb_run,
    stats_dict,
    current_step,
    step_offset=0,
    flush=False,
    skip_key_func=None,
    same_plot=False,
):
    def assert_scalar(value):
        assert is_number(value), "Only scalars are supported for tensorboard."

    def log_stat_func_wrapper(tb_run, nested_key_as_list, step):
        def log_stat_for_given_args(value):
            stat_name = ".".join(nested_key_as_list)
            assert_scalar(value)

            tb_run.add_scalar(stat_name, value, global_step=step)
            return value

        return log_stat_for_given_args

    def log_multiple_curves(tb_run, stats_dict, step):
        assert is_nested_dict(stats_dict)

        for plot_name, curves_dict in stats_dict.items():
            assert not is_nested_dict(curves_dict)

            for value in curves_dict.values():
                assert_scalar(value)

                tb_run.add_scalars(plot_name, curves_dict, global_step=step)

    step = current_step + step_offset

    if same_plot:
        log_multiple_curves(tb_run, stats_dict, step)

    else:
        dict_leaves_as_nested_keys = get_leaves_of_nested_dict(stats_dict)

        for nested_key_as_list in dict_leaves_as_nested_keys:
            assert len(nested_key_as_list)
            if skip_key_func is not None and skip_key_func(nested_key_as_list):
                continue

            apply_func_to_dict_by_nested_key(
                stats_dict,
                nested_key_as_list,
                func=log_stat_func_wrapper(tb_run, nested_key_as_list, step),
            )

    if flush:
        tb_run.flush()


def extract_id_from_spreadsheet_url(spreadsheet_url):
    return extract_by_regex_from_url(
        spreadsheet_url, [URL_KEY_RE, URL_SPREADSHEET_RE]
    )


def log_info(logger, tensor, name):
    if logger is None:
        logger = make_logger()
    logger.log(f"{name} norm: {torch.linalg.norm(tensor)}")
    cumsums = compute_tensor_cumsums(tensor)
    for dim_i in range(len(cumsums)):
        logger.log(
            f"{name} norm of cumsum for dim: {dim_i}: " f"{cumsums[dim_i]}"
        )
