"""
Core job submission and status management functionality.

This module handles the submission, monitoring, and status updates of jobs
in both local and SLURM environments.
"""
import multiprocessing as mp
import os
import re
import shlex
import signal
import subprocess
import time
from typing import Dict

import psutil
from stuned.job import Job, JobStatus, find_job_idx, get_slurm_job_status
from stuned.job_manager import JobManager
from stuned.utility.local_processing_utils import process_exists

# Global multiprocessing variables
# TODO: These should be passed as parameters rather than globals
run_jobs_flag = mp.Value("i", 1)  # 1 means jobs should run, 0 means they shouldn't
submitted_jobs = mp.Value("i", 0)
running_jobs = mp.Value("i", 0)
submitted_jobs_lock = mp.Lock()


def get_pid_job_stats(jobs_ids, logger):
    """Get job statistics for local processes by checking if PIDs exist."""
    job_stats_pid = {}
    for job_id in jobs_ids:
        # Just check if the process exists
        if process_exists(job_id):
            job_stats_pid[job_id] = JobStatus.RUNNING
        else:
            job_stats_pid[job_id] = JobStatus.FAILED  # I guess?
    return job_stats_pid


def get_slurm_jobs_stats(jobs_ids, logger):
    """Get job statistics from SLURM using sacct command."""
    # Convert the list of job IDs to a comma-separated string
    job_ids_str = ",".join(map(str, jobs_ids))

    # Define the sacct command with the desired format
    cmd = ["sacct", "-j", job_ids_str, "--format=JobID,State,ExitCode", "--noheader"]

    # Execute the command and capture the output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        logger.log("Warning: sacct command timed out! Will repeat after sleeping")
        return None
    # Split the output into lines and then into columns
    lines = result.stdout.strip().split("\n")
    job_stats = {}
    for line in lines:
        columns = line.split()
        # Check if the line contains the main job info (not the steps)
        if len(columns) == 3 and "+" not in columns[0]:
            job_id, state, exit_code = columns
            # check if str exists, extact ints only from job_id all ints not just sub
            job_id = int(re.sub("[^0-9]", "", job_id))
            job_stats[int(job_id)] = {"State": state, "ExitCode": exit_code}

    return job_stats


def update_job_statuses_in_place(
    job_manager: JobManager, shared_jobs_copy: Dict, logger, local_run=False
) -> bool:
    """Update job statuses in the job manager based on current system state."""
    # Check SLURM cluster for running jobs and their status
    ref_key_job_id = "main_pid" if local_run else "job_id"
    job_ids = [job[ref_key_job_id] for job in shared_jobs_copy.values()]

    if not local_run:
        # Check SLURM cluster for jobs and their status
        all_jobs_stats = get_slurm_jobs_stats(job_ids, logger)
        if all_jobs_stats is None:
            return False

        if all_jobs_stats == {}:
            return True
    else:
        # Status should be updated in the shared_jobs_copy itself
        pass

    # updated_rows = []
    for row_id, job in shared_jobs_copy.items():
        if not local_run:
            job_stat = all_jobs_stats.get(job[ref_key_job_id], None)
            if job_stat:
                slurm_status = job_stat["State"]
                exit_code = job_stat["ExitCode"]
                job_status = get_slurm_job_status(slurm_status)
            else:
                # Job not found in SLURM - check if we have an explicit status
                if "status" in job:
                    job_status = job["status"]
                    exit_code = job.get("exit_code", None)
                else:
                    # No SLURM status and no explicit status
                    slurm_status = None
                    exit_code = None
                    job_status = None
        else:
            # For local jobs, use the status from shared dict
            job_status = job.get("status", None)
            exit_code = job.get("exit_code", None)

        job_from_local_idx = find_job_idx(job_manager.jobs, job[ref_key_job_id])
        if job_from_local_idx is None:
            # Add a job to the list and mark as update
            job_to_add = Job(
                job_id=job[ref_key_job_id],
                job_status=job_status,
                job_exit_code=exit_code,
                csv_row_id=row_id,
            )
            job_to_add.updated = True
            job_manager.jobs.append(job_to_add)
            # updated_rows.append(row_id)
        else:
            job_from_local = job_manager.jobs[job_from_local_idx]
            if job_from_local.csv_row_id is None:
                job_from_local.csv_row_id = row_id
            if job_from_local.job_status != job_status or job_from_local.job_exit_code != exit_code:
                # Update the job and mark as update
                job_from_local.job_status = job_status
                job_from_local.job_exit_code = exit_code
                job_from_local.updated = True
                # updated_rows.append(row_id)
            job_manager.jobs[job_from_local_idx] = job_from_local
    return True


def update_job_status(process, shared_jobs_dict, row_id, lock_manager, cancelled=False):
    """Update the job status and exit code in the shared dictionary."""
    with lock_manager:
        # print(f"Before update: {shared_jobs_dict[row_id]}")
        job_data = shared_jobs_dict[row_id].copy()  # Get a local copy
        if cancelled:
            job_data["status"] = JobStatus.CANCELLED
        else:
            if process.returncode == 0:
                job_data["status"] = JobStatus.COMPLETED
            else:
                job_data["status"] = JobStatus.FAILED
        job_data["exit_code"] = process.returncode
        shared_jobs_dict[row_id] = job_data  # Set the modified data back
        # print(f"After update: {shared_jobs_dict[row_id]}")


def submit_job(
    run_cmd,
    log_file_path,
    run_locally,
    shared_jobs_dict,
    row_id,
    lock_manager,
    logger,
    max_conc_jobs=-1,
):
    """
    Submit a job for execution, either locally or via SLURM.
    
    Args:
        run_cmd: Command to execute
        log_file_path: Path to log file
        run_locally: Whether to run locally or via SLURM
        shared_jobs_dict: Shared dictionary for job tracking
        row_id: Row ID for tracking in spreadsheet
        lock_manager: Lock for thread safety
        logger: Logger instance
        max_conc_jobs: Maximum concurrent jobs (-1 for unlimited)
    
    Returns:
        Process return code or status string
    """
    if not run_jobs_flag.value:
        return "Job Stopped"

    with open(log_file_path, "w+") as log_file:
        if run_locally:
            def get_main_and_child_pids(pid):
                main_pid = pid
                child_pids = [child.pid for child in psutil.Process(main_pid).children()]
                return main_pid, child_pids

            while max_conc_jobs != -1 and running_jobs.value >= max_conc_jobs:
                if not run_jobs_flag.value:
                    return "Job Stopped"
                time.sleep(1)

            running_jobs.value += 1
            split_command = shlex.split(run_cmd)
            process = subprocess.Popen(split_command, stdout=log_file, stderr=log_file, shell=False)

            with submitted_jobs_lock:
                submitted_jobs.value += 1

            main_pid, child_pids = get_main_and_child_pids(process.pid)
            try:
                with lock_manager:
                    shared_jobs_dict[row_id] = {"main_pid": main_pid, "status": JobStatus.RUNNING}
            except Exception as e:
                pass
            # Periodically check if the process is still running
            while process.poll() is None:
                # Check the run_jobs_flag during execution
                if not run_jobs_flag.value:
                    logger.log(f"Stopping job with row ID: {row_id}")
                    # Send a SIGINT signal for graceful exit
                    process.send_signal(signal.SIGINT)

                    # Wait for up to 5 seconds for the process to exit gracefully
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # If the process doesn't exit within 5 seconds, terminate it forcefully
                        process.terminate()

                    # Update the job status after forcefully terminating
                    update_job_status(
                        process, shared_jobs_dict, row_id, lock_manager, cancelled=True
                    )
                    running_jobs.value -= 1
                    return 1

                time.sleep(1)  # Sleep for a short duration before checking again

            # Process finished
            update_job_status(process, shared_jobs_dict, row_id, lock_manager)
            running_jobs.value -= 1
            return process.returncode
        else:
            timeout_duration = 60

            try:
                if not run_jobs_flag.value:
                    return "Job Stopped"
                    
                output = subprocess.check_output(
                    run_cmd, stderr=subprocess.STDOUT, shell=True, timeout=timeout_duration
                ).decode("utf-8")
                numbers = re.findall(r"\d+", output)
                job_id = int("".join(numbers))

                if job_id:
                    logger.log(f"Submitted a job with ID: {job_id}")
                else:
                    logger.log("Failed to extract job ID from the output.")

            except subprocess.CalledProcessError as e:
                print(f"Command failed with error: {e.output.decode('utf-8')}")
                return 1
            except subprocess.TimeoutExpired:
                print(f"The command took longer than {timeout_duration} seconds to complete.")
                return 1

            with submitted_jobs_lock:
                submitted_jobs.value += 1

            with lock_manager:
                shared_jobs_dict[row_id] = {
                    "job_id": job_id,
                    "status": JobStatus.PENDING,  # Changed from "submitted" to use JobStatus enum
                }

            return 0


def get_all_slurm_jobs():
    """Get all SLURM jobs for the current user."""
    try:
        output = subprocess.check_output(["squeue", "--me"], universal_newlines=True, timeout=60)
        lines = output.strip().split("\n")[1:]  # Skip the header
        jobs = {}
        for line in lines:
            parts = line.split()
            job_id = int(parts[0])
            status = parts[4]
            jobs[job_id] = status
        return jobs
    except subprocess.CalledProcessError:
        return {}
    except subprocess.TimeoutExpired:
        return None 