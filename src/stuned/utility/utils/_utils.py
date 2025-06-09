import torch
import numpy as np
import sys
import os
from deepdiff import DeepDiff, model as dd_model
import pickle
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import io
import matplotlib.pyplot as plt
import gdown
from stnd.utility.utils import (
    PLT_COL_SIZE,
    PLT_ROW_SIZE,
    PLT_PLOT_HEIGHT,
    PLT_PLOT_WIDTH,
    TOL,
    prepare_for_unpickling,
    prepare_for_pickling,
    make_checkpoint_name,
    is_number,
    get_current_time,
    run_cmd_through_popen,
    get_filename_from_url,
    extract_tar_to_folder,
    remove_file_or_folder,
)


def compute_dicts_diff(dict1, dict2, ignore_order=True):
    ddiff = DeepDiff(dict1, dict2, ignore_order=ignore_order)
    return ddiff


def randomly_subsample_indices_uniformly(total_samples, num_to_subsample):
    weights = torch.tensor(
        total_samples * [1.0 / total_samples], dtype=torch.float
    )
    return torch.multinomial(weights, num_to_subsample)


def deterministically_subsample_indices_uniformly(
    total_samples, num_to_subsample
):
    assert (
        num_to_subsample <= total_samples
    ), "Try to subsample more samples than exist."
    return torch.linspace(
        0, total_samples - 1, num_to_subsample, dtype=torch.int
    )


def get_device(use_gpu, idx=0):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda:{}".format(idx))
        else:
            raise Exception("Cuda is not available.")
    else:
        return torch.device("cpu")


def get_model_device(model):
    # if timm model
    if hasattr(model, "blocks"):
        model = model.blocks

    assert isinstance(model, torch.nn.Module)
    return next(model.parameters()).device


def read_checkpoint(checkpoint_path, map_location=None):
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
            else:
                return super().find_class(module, name)

    def do_read_checkpoint(file, map_location=None):
        if map_location == "cpu" or map_location == torch.device(type="cpu"):
            checkpoint = CPU_Unpickler(file).load()
        else:
            try:
                checkpoint = torch.load(file, map_location=map_location)
            except:
                checkpoint = pickle.load(file)

        return checkpoint

    if os.path.exists(checkpoint_path):
        file = open(checkpoint_path, "rb")
        try:
            checkpoint = do_read_checkpoint(file, map_location=map_location)
        except RuntimeError:
            checkpoint = do_read_checkpoint(file, map_location="cpu")
    else:
        raise Exception(
            "Checkpoint path does not exist: {}".format(checkpoint_path)
        )
    for obj in checkpoint.values():
        prepare_for_unpickling(obj)
    return checkpoint


def save_checkpoint(
    checkpoint, checkpoint_folder, checkpoint_name=None, logger=None
):
    if checkpoint_name is None:
        checkpoint_name = make_checkpoint_name(checkpoint)
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_savepath = os.path.join(checkpoint_folder, checkpoint_name)
    log_msg = "Saving checkpoint to {}".format(checkpoint_savepath)
    if logger:
        logger.log(log_msg, auto_newline=True)
    else:
        print(log_msg)
    for obj in checkpoint.values():
        prepare_for_pickling(obj)
    torch.save(checkpoint, open(checkpoint_savepath, "wb"))
    for obj in checkpoint.values():
        prepare_for_unpickling(obj)
    return checkpoint_savepath


def assert_two_values_are_close(value_1, value_2, **isclose_kwargs):
    def assert_info(value_1, value_2):
        return "value 1: {}\n\nvalue 2: {}".format(value_1, value_2)

    def assert_is_close(value_1, value_2, isclose_func, **isclose_kwargs):
        assert isclose_func(
            value_1, value_2, **isclose_kwargs
        ).all(), assert_info(value_1, value_2)

    if value_1 is None:
        assert value_2 is None

    if not (is_number(value_1) and is_number(value_2)):
        value_1_type = type(value_1)
        value_2_type = type(value_2)
        assert value_1_type == value_2_type, assert_info(
            value_1_type, value_2_type
        )

    if isinstance(value_1, (list, tuple, dict)):
        assert len(value_1) == len(value_2), assert_info(value_1, value_2)
        if isinstance(value_1, dict):
            iterable = zip(
                sorted(value_1.items(), key=(lambda x: x[0])),
                sorted(value_2.items(), key=(lambda x: x[0])),
            )
        else:
            iterable = zip(value_1, value_2)
        for subvalue_1, subvalue_2 in iterable:
            assert_two_values_are_close(
                subvalue_1, subvalue_2, **isclose_kwargs
            )
    elif isinstance(value_1, np.ndarray):
        assert_is_close(value_1, value_2, np.isclose, **isclose_kwargs)
    elif torch.is_tensor(value_1):
        assert_is_close(value_1, value_2, torch.isclose, **isclose_kwargs)
    else:
        assert value_1 == value_2, assert_info(value_1, value_2)


def cat_or_assign(accumulating_tensor, new_tensor):
    if accumulating_tensor is None:
        return new_tensor
    return torch.cat((accumulating_tensor, new_tensor))


def read_old_checkpoint(checkpoint_path, map_location=None):
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_eval"),
    )
    checkpoint = read_checkpoint(checkpoint_path, map_location=map_location)
    sys.path.pop(0)
    return checkpoint


def read_model_from_old_checkpoint(path):
    checkpoint = read_old_checkpoint(path)
    return checkpoint["model"]


class TimeStampEventHandler(FileSystemEventHandler):
    def __init__(self):
        super(TimeStampEventHandler, self).__init__()
        self.update_time()

    def on_any_event(self, event):
        self.update_time()

    def update_time(self):
        self.last_change_time = get_current_time()

    def has_events(self, delta):
        time_since_last_change = (
            get_current_time() - self.last_change_time
        ).total_seconds()
        if time_since_last_change <= delta:
            return True
        else:
            return False


def folder_still_has_updates(path, delta, max_time, check_time=None):
    """
    Check every <check_time> seconds whether <path> had any updates (events).
    Observe the <path> for at most <max_time>.
    If there were no updates for <delta> seconds return True, otherwise return
    False. If watchdog observer failed to start return None.
    """

    if check_time is None:
        check_time = delta

    n = max(1, int(max_time / check_time))
    i = 0
    event_handler = TimeStampEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    try:
        observer.start()
    except:
        return None

    has_events_bool = event_handler.has_events(delta)
    while has_events_bool and i < n:
        time.sleep(check_time)
        i += 1
        has_events_bool = event_handler.has_events(delta)

    observer.stop()
    observer.join()

    return has_events_bool


def aggregate_tensors_by_func(input_list, func=torch.mean):
    return func(torch.stack(input_list))


def show_images(images, label_lists=None):
    def remove_ticks_and_labels(subplot):
        subplot.axes.xaxis.set_ticklabels([])
        subplot.axes.yaxis.set_ticklabels([])
        subplot.axes.xaxis.set_visible(False)
        subplot.axes.yaxis.set_visible(False)

    def get_row_cols(n):
        n_rows = int(np.sqrt(n))
        n_cols = int(n / n_rows)
        if n % n_rows != 0:
            n_cols += 1
        return n_rows, n_cols

    n = len(images)
    assert n > 0
    if label_lists is not None:
        for label_list in label_lists.values():
            assert len(label_list) == n

    n_rows, n_cols = get_row_cols(n)

    cmap = get_cmap(images[0])
    fig = plt.figure(figsize=(n_cols * PLT_COL_SIZE, n_rows * PLT_ROW_SIZE))
    for i in range(n):
        subplot = fig.add_subplot(n_rows, n_cols, i + 1)
        title = f"n{i}"
        if label_lists is not None:
            for label_name, label_list in label_lists.items():
                title += f'\n{label_name}="{label_list[i]}"'
        subplot.title.set_text(title)
        remove_ticks_and_labels(subplot)

        imshow(subplot, images[i], cmap=cmap)

    plt.tight_layout()
    plt.show()


def imshow(plot, image, cmap=None, color_dim_first=True):
    image = image.squeeze()
    num_image_dims = len(image.shape)
    if cmap is None:
        cmap = get_cmap(image)
    assert num_image_dims == 2 or num_image_dims == 3
    if num_image_dims == 3 and color_dim_first:
        image = np.transpose(image, (1, 2, 0))
    plot.imshow(image, cmap=cmap)


def get_cmap(image):
    cmap = "viridis"
    squeezed_shape = image.squeeze().shape
    if len(squeezed_shape) == 2:
        cmap = "gray"
    return cmap


def compute_tensor_cumsums(tensor):
    result = []
    for dim_i in range(len(tensor.shape)):
        result.append(torch.linalg.norm(torch.cumsum(tensor, dim=dim_i)))
    return result


def compute_unique_tensor_value(tensor):
    return torch.round(
        TOL * aggregate_tensors_by_func(compute_tensor_cumsums(tensor))
    )


def download_file(file_path, download_url):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if "google" in download_url:
            download_type = "gdrive"
        else:
            download_type = "wget"

        if download_type == "wget":
            run_cmd_through_popen(
                f"wget {download_url} -O {file_path}", logger=None
            )
        else:
            assert download_type == "gdrive"
            assert "uc?id=" in download_url, (
                "When using gdown, url should be of form: "
                "https://drive.google.com/uc?id=<file_id>"
            )

            gdown.download(
                download_url,
                file_path,
                quiet=False,
                use_cookies=False,
                fuzzy=True,
            )


def download_and_extract_tar(
    parent_folder, download_url, name=None, extension=None
):
    if name is None:
        cur_time = str(get_current_time()).replace(" ", "_")
        name = f"tmp_tar_{cur_time}"
    if extension is None:
        original_filename = get_filename_from_url(download_url)
        if original_filename is not None and "." in original_filename:
            before, dot, after = original_filename.partition(".")
            extension = dot + after
        else:
            extension = ".tar.gz"
    downloaded_tar = os.path.join(parent_folder, f"{name}{extension}")
    download_file(downloaded_tar, download_url)
    extract_tar_to_folder(downloaded_tar, parent_folder)
    remove_file_or_folder(downloaded_tar)


def optionally_make_dir(path, call_dirname=True):
    if call_dirname:
        base_dir = os.path.dirname(path)
    else:
        base_dir = path
    if base_dir != "":
        os.makedirs(base_dir, exist_ok=True)
