"""
SLURM Integration Module

This module handles:
- SLURM batch script generation and configuration
- SLURM arguments processing and validation
- SLURM job configuration from CSV parameters
- Cluster-specific SLURM settings and defaults
"""

import os
import copy

from stuned.utility.logger import (
    SLURM_PREFIX,
    PREFIX_SEPARATOR,
    PLACEHOLDERS_FOR_DEFAULT,
)

# Constants from main module that should eventually be in a constants module
DEV_NULL = "/dev/null"
DEFAULT_SLURM_ARGS_DICT = {
    # "partition": "gpu-2080ti-beegfs",
    "gpus": 1,
    "time": "24:00:00",
    "ntasks": 1,
    "cpus-per-task": 2,
    "error": DEV_NULL,
    "output": DEV_NULL,
}


def fill_sbatch_script(sbatch_file, slurm_args_dict, command):
    """
    Fill SLURM batch script with headers and command.
    
    Creates a complete SLURM batch script by writing the shebang line,
    all SLURM directives, and the command to execute.
    
    Args:
        sbatch_file: File object to write the script to
        slurm_args_dict: Dictionary of SLURM arguments and their values
        command: The command string to execute
    """
    sbatch_file.write("#!/bin/bash\n")
    for slurm_arg, value in slurm_args_dict.items():
        sbatch_file.write("#SBATCH --{}={}\n".format(slurm_arg, value))
    sbatch_file.write(command)
    sbatch_file.flush()


def make_slurm_args_dict(csv_row, exp_name, log_file):
    """
    Create SLURM arguments dictionary from CSV row and experiment configuration.
    
    Combines default SLURM settings with experiment-specific parameters
    and CSV-specified SLURM arguments.
    
    Args:
        csv_row: Dictionary representing a single CSV row with potential SLURM parameters
        exp_name: Experiment name for job identification
        log_file: Path for SLURM output and error logs
        
    Returns:
        dict: Complete SLURM arguments dictionary ready for job submission
    """
    # Import here to avoid circular imports
    from stuned.run_from_csv.config_processor import extract_from_csv_row_by_prefix
    
    # Start with default SLURM configuration
    all_slurm_args_dict = copy.deepcopy(DEFAULT_SLURM_ARGS_DICT)
    
    # Set job-specific parameters
    all_slurm_args_dict["job-name"] = exp_name
    all_slurm_args_dict["output"] = log_file
    all_slurm_args_dict["error"] = log_file

    # Extract SLURM-specific arguments from CSV row
    specified_slurm_args = extract_from_csv_row_by_prefix(
        csv_row, SLURM_PREFIX + PREFIX_SEPARATOR, ignore_values=PLACEHOLDERS_FOR_DEFAULT
    )
    
    # Merge CSV-specified SLURM args with defaults (CSV args take precedence)
    all_slurm_args_dict |= specified_slurm_args

    # Ensure log directories exist
    os.makedirs(os.path.dirname(all_slurm_args_dict["output"]), exist_ok=True)
    os.makedirs(os.path.dirname(all_slurm_args_dict["error"]), exist_ok=True)
    
    return all_slurm_args_dict


def validate_slurm_args(slurm_args_dict):
    """
    Validate SLURM arguments dictionary for common issues.
    
    Args:
        slurm_args_dict: Dictionary of SLURM arguments
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If critical SLURM configuration issues are found
    """
    required_args = ["job-name", "output", "error"]
    
    for arg in required_args:
        if arg not in slurm_args_dict:
            raise ValueError(f"Required SLURM argument '{arg}' is missing")
    
    # Validate time format (basic check)
    if "time" in slurm_args_dict:
        time_str = slurm_args_dict["time"]
        if not isinstance(time_str, str) or ":" not in time_str:
            raise ValueError(f"Invalid time format: {time_str}. Expected format like '24:00:00'")
    
    # Validate numeric parameters
    numeric_args = ["gpus", "ntasks", "cpus-per-task"]
    for arg in numeric_args:
        if arg in slurm_args_dict:
            try:
                int(slurm_args_dict[arg])
            except (ValueError, TypeError):
                raise ValueError(f"SLURM argument '{arg}' must be numeric, got: {slurm_args_dict[arg]}")
    
    return True


def get_cluster_specific_defaults(cluster_region=None):
    """
    Get cluster-specific SLURM defaults based on the detected region.
    
    Args:
        cluster_region: The detected cluster region (from ai_center_cluster_specific)
        
    Returns:
        dict: Cluster-specific SLURM default arguments
    """
    defaults = copy.deepcopy(DEFAULT_SLURM_ARGS_DICT)
    
    # Add cluster-specific configurations here as needed
    # This can be extended based on the cluster detection logic
    if cluster_region:
        # Example: different defaults for different clusters
        # if cluster_region == CLUSTER.GALVANI:
        #     defaults["partition"] = "gpu-2080ti-beegfs"
        # elif cluster_region == CLUSTER.FERRANTI:
        #     defaults["partition"] = "gpu-a100-beegfs"
        pass
    
    return defaults


def format_slurm_command_args(slurm_args_dict):
    """
    Format SLURM arguments dictionary into command-line argument string.
    
    Useful for srun commands where arguments need to be passed as command-line flags.
    
    Args:
        slurm_args_dict: Dictionary of SLURM arguments
        
    Returns:
        str: Formatted command-line arguments string
    """
    return " ".join([f"--{flag}={value}" for flag, value in slurm_args_dict.items()])


def extract_slurm_job_id_from_output(sbatch_output):
    """
    Extract SLURM job ID from sbatch command output.
    
    Args:
        sbatch_output: String output from sbatch command
        
    Returns:
        str or None: Extracted job ID, or None if not found
    """
    import re
    
    # sbatch typically outputs: "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', sbatch_output)
    if match:
        return match.group(1)
    
    return None 