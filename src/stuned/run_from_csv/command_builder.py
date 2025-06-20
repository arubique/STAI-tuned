"""
Command Generation Module

This module handles:
- Task command building and configuration
- SLURM command generation and wrapping
- CSV column validation for command processing
- Command argument processing and validation
"""

import os
from tempfile import NamedTemporaryFile

from stuned.utility.utils import NEW_SHELL_INIT_COMMAND
from stuned.utility.logger import (
    SLURM_PREFIX,
    PREFIX_SEPARATOR,
    PLACEHOLDERS_FOR_DEFAULT,
)

# Constants from main module that should eventually be in a constants module
PATH_TO_DEFAULT_CONFIG_COLUMN = "path_to_default_config"
MAIN_PATH_COLUMN = "path_to_main"
WHETHER_TO_RUN_COLUMN = "whether_to_run"
USE_SRUN = False


def make_task_cmd(new_config_path, conda_env, exec_path, csv_row):
    """
    Create the task command string for execution.
    
    Args:
        new_config_path: Path to the generated configuration file
        conda_env: Conda environment name to use
        exec_path: Path to the main execution script
        csv_row: CSV row data containing potential custom run command
        
    Returns:
        str: Complete command string ready for execution
    """
    exec_args = "--config_path {}".format(new_config_path)

    # If `custom_run_cmd` is passed, overwrite the default command
    # with the custom one.
    if "custom_run_cmd" in csv_row:
        cmd = "{} {} {}".format(csv_row["custom_run_cmd"], exec_path, exec_args)
    else:
        cmd = "{} {} && python {} {}".format(
            NEW_SHELL_INIT_COMMAND, conda_env, exec_path, exec_args
        )
    return cmd


def make_final_cmd_slurm(csv_row, exp_name, log_file_path, cmd_as_string):
    """
    Create SLURM command for job submission.
    
    Takes a basic command string and wraps it in SLURM submission logic,
    either using srun or sbatch depending on configuration.
    
    Args:
        csv_row: CSV row containing SLURM configuration parameters
        exp_name: Experiment name for job identification
        log_file_path: Path for SLURM log files
        cmd_as_string: The basic command to wrap in SLURM
        
    Returns:
        str: Complete SLURM submission command
    """
    # Import here to avoid circular imports
    from stuned.run_from_csv.slurm_utils import make_slurm_args_dict, fill_sbatch_script
    
    slurm_args_dict = make_slurm_args_dict(csv_row, exp_name, log_file_path)
    
    if USE_SRUN:
        slurm_args_as_string = " ".join(
            [f"--{flag}={value}" for flag, value in slurm_args_dict.items()]
        )
        final_cmd = 'srun {} sh -c "{}" &'.format(slurm_args_as_string, cmd_as_string)
    else:
        with NamedTemporaryFile("w", delete=False) as tmp_file:
            fill_sbatch_script(tmp_file, slurm_args_dict, cmd_as_string)
            final_cmd = "sbatch {}".format(tmp_file.name)

    return final_cmd


def check_csv_column_names(csv_row, allowed_prefixes):
    """
    Validate CSV column names against allowed prefixes and required columns.
    
    Ensures that the CSV has all required columns and that any prefixed
    columns use valid prefixes for command generation.
    
    Args:
        csv_row: Dictionary representing a single CSV row
        allowed_prefixes: List of allowed column prefixes (e.g., 'slurm:', 'delta:')
        
    Raises:
        AssertionError: If required columns are missing or invalid prefixes are used
    """
    # Check for required columns
    assert MAIN_PATH_COLUMN in csv_row, f"Required column '{MAIN_PATH_COLUMN}' missing from CSV"
    assert WHETHER_TO_RUN_COLUMN in csv_row, f"Required column '{WHETHER_TO_RUN_COLUMN}' missing from CSV"
    assert PATH_TO_DEFAULT_CONFIG_COLUMN in csv_row, f"Required column '{PATH_TO_DEFAULT_CONFIG_COLUMN}' missing from CSV"

    # Validate column names and prefixes
    for i, key in enumerate(csv_row.keys()):
        assert key is not None, (
            f"Column {i} has empty column name. " f"Or some table entries contain commas."
        )
        if PREFIX_SEPARATOR in key:
            assert any([prefix in key for prefix in allowed_prefixes]), (
                f'"{key}" does not contain any of allowed prefixes ' f"from:\n{allowed_prefixes}\n"
            )


def validate_command_requirements(csv_row):
    """
    Validate that a CSV row has all requirements for command generation.
    
    Args:
        csv_row: Dictionary representing a single CSV row
        
    Returns:
        bool: True if row is valid for command generation
        
    Raises:
        ValueError: If critical command requirements are not met
    """
    # Check that main path exists if specified
    if MAIN_PATH_COLUMN in csv_row and csv_row[MAIN_PATH_COLUMN]:
        main_path = csv_row[MAIN_PATH_COLUMN]
        if not os.path.exists(main_path):
            raise ValueError(f"Main execution path does not exist: {main_path}")
    
    # Check whether_to_run is valid
    if WHETHER_TO_RUN_COLUMN in csv_row:
        whether_to_run = csv_row[WHETHER_TO_RUN_COLUMN]
        if not whether_to_run.isnumeric():
            raise ValueError(f"'{WHETHER_TO_RUN_COLUMN}' must be numeric, got: {whether_to_run}")
    
    return True


def extract_custom_command_args(csv_row):
    """
    Extract custom command arguments from CSV row.
    
    Args:
        csv_row: Dictionary representing a single CSV row
        
    Returns:
        dict: Dictionary of custom command arguments and their values
    """
    custom_args = {}
    
    # Look for custom_run_cmd
    if "custom_run_cmd" in csv_row and csv_row["custom_run_cmd"]:
        custom_args["custom_run_cmd"] = csv_row["custom_run_cmd"]
    
    # Extract any other command-related custom parameters
    command_prefixes = ["cmd_", "command_", "exec_"]
    for key, value in csv_row.items():
        if any(key.startswith(prefix) for prefix in command_prefixes):
            custom_args[key] = value
    
    return custom_args 