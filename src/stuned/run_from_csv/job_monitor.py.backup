"""
Job monitoring and orchestration functionality.

This module handles the complex orchestration of job monitoring, progress tracking,
signal handling, and graceful cleanup for both local and SLURM environments.
"""
import datetime
import os
import signal
import subprocess
import sys
import time
from warnings import warn

from stuned.gsheet_batch_updater import GSheetBatchUpdater
from stuned.job import JobStatus
from stuned.job_manager import JobManager
from stuned.utility.local_processing_utils import process_exists
from stuned.utility.logger import GspreadClient

# Import from job_submission for shared variables and functions
from stuned.run_from_csv.job_submission import (
    run_jobs_flag,
    submitted_jobs,
    running_jobs,
    update_job_statuses_in_place,
    submit_job,
)

# Constants from main module - should be imported from constants module eventually
MONITOR_STATUS_COLUMN = "status_monitor"
MONITOR_EXIT_CODE_COLUMN = "exit_code_monitor"
MONITOR_JOB_ID_COLUMN = "slurm_job_id_monitor"
MONITOR_LAST_UPDATE_COLUMN = "last_update_monitor"


def dump_into_gsheet_queue(gsheet_updater, job_manager: JobManager):
    """
    Update the Google Sheets queue with job status information.
    
    Args:
        gsheet_updater: GSheetBatchUpdater instance
        job_manager: JobManager instance containing job status information
    """
    for idx, job in enumerate(job_manager.jobs):
        if job.updated:
            gsheet_updater.add_to_queue(job.csv_row_id, MONITOR_STATUS_COLUMN, job.job_status)
            gsheet_updater.add_to_queue(
                job.csv_row_id,
                MONITOR_LAST_UPDATE_COLUMN,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            if job.job_exit_code is not None:
                gsheet_updater.add_to_queue(
                    job.csv_row_id, MONITOR_EXIT_CODE_COLUMN, f"{job.job_exit_code}"
                )
            if job.job_id is not None:
                gsheet_updater.add_to_queue(job.csv_row_id, MONITOR_JOB_ID_COLUMN, f"{job.job_id}")

            # check the queue of each job too
            for q_item in job.writing_queue:
                gsheet_updater.add_to_queue(job.csv_row_id, q_item[0], q_item[1])
            job.writing_queue = []
            job.updated = False
            job_manager.jobs[idx] = job


def monitor_jobs_async(
    job_manager,
    pool,
    job_submission_args,
    shared_jobs_dict,
    run_locally: bool,
    logger,
    spreadsheet_url,
    worksheet_name: str,
    shared_row_numbers,
    csv_path,
    gsheet_client,
    lock,
    n_jobs_total,
    input_csv,
    max_conc_jobs=-1,
):
    """
    Asynchronously monitor and manage job execution.
    
    This is the main orchestration function that handles:
    - Job submission with concurrency limits
    - Real-time status monitoring
    - Progress reporting
    - Signal handling and graceful cleanup
    - Google Sheets integration
    """
    # Import constants that should eventually be in a constants module
    from stuned.utility.constants import SUBMITTED_STATUS, WHETHER_TO_RUN_COLUMN
    from stuned.utility.logger import SLURM_PREFIX, DELTA_PREFIX
    
    # Initialize counters
    submitted = 0
    running = 0
    finished_successfully = 0
    failed = 0
    
    # Define a flag to check if we should exit
    should_exit = False
    signal_received = False  # Track if we've already received a signal
    is_cancelling = False   # Track if we're in the process of cancelling

    # Track which jobs are pending submission
    pending_jobs = list(range(len(job_submission_args)))
    active_submissions = {}  # job_idx -> AsyncResult
    submitted_not_running = set()  # Track jobs that are submitted but not yet running

    # Signal handler function
    def signal_handler(sig, frame):
        nonlocal should_exit, signal_received, is_cancelling
        if is_cancelling:
            # If we're already cancelling, ignore additional signals
            logger.log("Cancellation in progress, please wait...")
            return
        if not signal_received:
            signal_received = True
            run_jobs_flag.value = 0  # This will prevent new jobs from being submitted
            logger.log("Received exit signal. Preparing to terminate all jobs...")
            should_exit = True
        # If we get additional signals, just ignore them and let the cancellation complete

    # Register the signal handler
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        gsheet_updater = None
        shared_row_numbers_lst = list(shared_row_numbers)

        if csv_path is None:
            assert run_locally, "If not running locally, a csv file must be provided."
            warn("No csv file provided, skipping writing to a csv file.")
        else:
            gsheet_updater = GSheetBatchUpdater(
                spreadsheet_url, worksheet_name, gsheet_client, logger, csv_path, input_csv
            )
            # get cols from gsheet_client using the `worksheet_name`
            worksheet = gsheet_client.opened_spreadsheet.worksheet(worksheet_name)
            # Get the first row (headers)
            headers_org = worksheet.row_values(1)

            # Prepare the updates
            cols_not_reset = ["path_to_main", "path_to_default_config", "custom_run_cmd"]

            # also exclude slurm and delta columns
            cols_not_reset += [
                col
                for col in headers_org
                if col.startswith(SLURM_PREFIX) or col.startswith(DELTA_PREFIX)
            ]

            headers_reset = [col for col in headers_org if col not in cols_not_reset]

            column_value_pairs = [
                (MONITOR_LAST_UPDATE_COLUMN, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                (MONITOR_STATUS_COLUMN, SUBMITTED_STATUS),
                (WHETHER_TO_RUN_COLUMN, "0"),
            ]
            for column_name in headers_reset:
                if column_name != WHETHER_TO_RUN_COLUMN:
                    column_value_pairs.append((column_name, ""))

            # Add updates to the queue
            for row_id_share in shared_row_numbers_lst:
                for key, val in column_value_pairs:
                    gsheet_updater.add_to_queue(row_id_share, key, val)

            # Write updates to CSV and then batch update the Google Sheet
            update_status = gsheet_updater.batch_update()
            logger.log(f"Update status to GSheets: {update_status}")

        total_jobs = n_jobs_total
        if total_jobs == 0:
            logger.log("No jobs submitted, exiting.")
            return

        current_sleep_duration = 1
        sleep_increment = 1
        max_sleep_duration = 15

        # Main monitoring loop
        while not should_exit:
            # Check current running jobs and pending submissions
            current_jobs = 0
            if not run_locally:
                try:
                    # Get all jobs and their statuses
                    active_job_ids = set()
                    with lock:
                        shared_jobs_copy = dict(shared_jobs_dict)
                        for job_info in shared_jobs_copy.values():
                            if 'job_id' in job_info and 'status' in job_info:
                                status = job_info['status']
                                # Only count jobs that are actually running or pending
                                if status in [JobStatus.RUNNING, JobStatus.PENDING]:
                                    active_job_ids.add(job_info['job_id'])
                                    
                    # Also check SLURM queue to verify jobs are actually running
                    try:
                        output = subprocess.check_output(["squeue", "--me", "--noheader"], universal_newlines=True)
                        slurm_jobs = set(line.split()[0] for line in output.strip().split('\n') if line.strip())
                        # Only keep jobs that are both in our tracking AND in SLURM queue
                        active_job_ids = {str(job_id) for job_id in active_job_ids if str(job_id) in slurm_jobs}
                    except subprocess.CalledProcessError:
                        pass  # If squeue fails, fall back to our tracking
                    
                    # Count only jobs that are verified to be running
                    current_jobs = len(active_job_ids) + len(active_submissions)
                    logger.log(f"DEBUG: Found {current_jobs} active jobs (verified in SLURM)")
                except Exception as e:
                    logger.log(f"Failed to check job status: {str(e)}")
                    time.sleep(5)
                    continue
            else:
                current_jobs = running_jobs.value

            # Submit new jobs if under limit
            logger.log(f"DEBUG: Before submission - current_jobs: {current_jobs}, active_submissions: {len(active_submissions)}, max_conc_jobs: {max_conc_jobs}")
            while pending_jobs and (max_conc_jobs == -1 or current_jobs < max_conc_jobs):
                job_idx = pending_jobs.pop(0)
                args_submit = job_submission_args[job_idx]
                logger.log(f"DEBUG: Submitting job {job_idx}")
                active_submissions[job_idx] = pool.apply_async(submit_job, args_submit)
                current_jobs += 1
                logger.log(f"Submitting job {job_idx}, current jobs: {current_jobs}/{max_conc_jobs}")

            # Check completed submissions
            completed = []
            for job_idx, async_result in active_submissions.items():
                if async_result.ready():
                    try:
                        result = async_result.get(timeout=1)
                        logger.log(f"DEBUG: Job {job_idx} submission completed with result {result}")
                        if result != 0:
                            logger.log(f"Job submission failed for index {job_idx}")
                            current_jobs -= 1  # Decrement since submission failed
                            pending_jobs.append(job_idx)
                    except Exception as e:
                        logger.log(f"Error checking job {job_idx}: {str(e)}")
                        current_jobs -= 1  # Decrement since submission failed
                        pending_jobs.append(job_idx)
                    completed.append(job_idx)

            # Remove completed submissions
            for job_idx in completed:
                del active_submissions[job_idx]

            # Wait for job manager to register the jobs and update statuses
            if not run_locally:
                with job_manager.manager_lock:
                    jobs_update_status = update_job_statuses_in_place(
                        job_manager, shared_jobs_dict, logger, run_locally
                    )
                    if jobs_update_status and gsheet_updater is not None:
                        dump_into_gsheet_queue(gsheet_updater, job_manager)

            # Count the statuses
            submitted = submitted_jobs.value
            running = sum(1 for job in job_manager.jobs if job.job_status == JobStatus.RUNNING)
            finished_successfully = sum(1 for job in job_manager.jobs if job.job_status == JobStatus.COMPLETED)
            failed = sum(
                1
                for job in job_manager.jobs
                if job.job_status in [JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]
            )

            # Display progress and status
            last_update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_msg = (
                f"Jobs Progress | Submitted: {submitted}/{total_jobs} | "
                f"Running: {running}/{total_jobs} | "
                f"Finished: {finished_successfully}/{total_jobs} | "
                f"Failed: {failed} | "
                f"Pending: {len(pending_jobs)} | "
                f"Active Submissions: {len(active_submissions)} | "
                f"Last Update: {last_update_time}"
            )
            logger.log(status_msg, carriage_return=True)

            if gsheet_updater is not None:
                gsheet_updater.batch_update()

            time.sleep(current_sleep_duration)
            current_sleep_duration = min(current_sleep_duration + sleep_increment, max_sleep_duration)

            if finished_successfully + failed == total_jobs:
                logger.log("All jobs finished.")
                break

    finally:
        if should_exit:
            is_cancelling = True  # Mark that we're in cancellation process
            # Save original handlers
            original_sigint = signal.getsignal(signal.SIGINT)
            original_sigterm = signal.getsignal(signal.SIGTERM)
            
            # Block all signals during critical operations
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            
            try:
                logger.log("Cancelling pending submissions...")
                for job_idx in list(active_submissions.keys()):
                    async_result = active_submissions[job_idx]
                    if not async_result.ready():
                        # Mark as cancelled in shared dict
                        with lock:
                            row_id = job_submission_args[job_idx][-3]  # Get row_id from args
                            if row_id in shared_jobs_dict:
                                job_info = shared_jobs_dict[row_id].copy()
                                job_info['status'] = JobStatus.CANCELLED
                                shared_jobs_dict[row_id] = job_info
                
                # Update status for pending jobs before proceeding
                with lock:
                    shared_jobs_copy = dict(shared_jobs_dict)
                with job_manager.manager_lock:
                    jobs_update_status = update_job_statuses_in_place(
                        job_manager, shared_jobs_copy, logger, run_locally
                    )
                    if jobs_update_status and gsheet_updater is not None:
                        dump_into_gsheet_queue(gsheet_updater, job_manager)
                        gsheet_updater.batch_update(force=True)

                logger.log("Cancelling running jobs...")
                running_job_ids = []
                with lock:
                    shared_jobs_copy = dict(shared_jobs_dict)
                    for row_id, job_info in shared_jobs_copy.items():
                        if 'job_id' in job_info:
                            running_job_ids.append((job_info['job_id'], row_id))
                            # Update status in shared dict
                            job_info['status'] = JobStatus.CANCELLED
                            shared_jobs_dict[row_id] = job_info
                        elif 'main_pid' in job_info:
                            running_job_ids.append((job_info['main_pid'], row_id))
                            # Update status in shared dict
                            job_info['status'] = JobStatus.CANCELLED
                            shared_jobs_dict[row_id] = job_info

                if running_job_ids:
                    logger.log(f"Sending cancellation to {len(running_job_ids)} jobs...")
                    # Split into SLURM and local jobs
                    slurm_jobs = []
                    local_jobs = []
                    for job_id, row_id in running_job_ids:
                        if run_locally:
                            local_jobs.append(job_id)
                        else:
                            slurm_jobs.append(str(job_id))

                    # Cancel all SLURM jobs at once
                    if slurm_jobs:
                        logger.log("Cancelling SLURM jobs...")
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                # First try normal cancel
                                subprocess.run(["scancel", *slurm_jobs], check=True)
                                
                                # Wait and verify jobs are gone
                                remaining_jobs = []
                                for _ in range(5):  # Check multiple times with delay
                                    remaining_jobs = []
                                    try:
                                        output = subprocess.check_output(["squeue", "--me", "--noheader"], universal_newlines=True)
                                        for line in output.strip().split('\n'):
                                            if line.strip():
                                                job_id = line.split()[0]
                                                if job_id in slurm_jobs:
                                                    remaining_jobs.append(job_id)
                                    except subprocess.CalledProcessError:
                                        break  # No jobs in queue is good
                                    
                                    if not remaining_jobs:
                                        break
                                    logger.log(f"Waiting for {len(remaining_jobs)} jobs to cancel...")
                                    time.sleep(2)
                                
                                if not remaining_jobs:
                                    logger.log("All SLURM jobs cancelled successfully")
                                    break
                                
                                # If jobs still exist, try force kill
                                if retry < max_retries - 1:
                                    logger.log(f"Some jobs still running, trying force kill (attempt {retry + 1}/{max_retries})...")
                                    subprocess.run(["scancel", "--signal=KILL", *remaining_jobs], check=True)
                                else:
                                    logger.log(f"WARNING: Failed to cancel jobs after {max_retries} attempts: {remaining_jobs}")
                                    
                            except Exception as e:
                                logger.log(f"Error during cancellation attempt {retry + 1}: {str(e)}")
                                if retry == max_retries - 1:
                                    logger.log("Failed to cancel all jobs, continuing with cleanup...")

                    # Kill local jobs one by one (we have to since they're processes)
                    for job_id in local_jobs:
                        if process_exists(job_id):
                            try:
                                os.kill(job_id, signal.SIGKILL)
                            except:
                                pass

                # Make sure all jobs are marked as cancelled in job manager
                logger.log("Updating final status in sheets...")
                try:
                    with lock:
                        shared_jobs_copy = dict(shared_jobs_dict)
                    with job_manager.manager_lock:
                        # First ensure all jobs are marked as cancelled in job manager
                        for job in job_manager.jobs:
                            if job.job_status in [JobStatus.RUNNING, JobStatus.PENDING]:
                                job.job_status = JobStatus.CANCELLED
                                job.updated = True
                                # Also update in shared dict to ensure consistency
                                if job.csv_row_id in shared_jobs_dict:
                                    job_info = shared_jobs_dict[job.csv_row_id].copy()
                                    job_info['status'] = JobStatus.CANCELLED
                                    shared_jobs_dict[job.csv_row_id] = job_info
                        
                        # Force a final status update
                        jobs_update_status = update_job_statuses_in_place(
                            job_manager, shared_jobs_copy, logger, run_locally
                        )

                        # Make sure all cancelled jobs are marked for update
                        for job in job_manager.jobs:
                            if job.job_status == JobStatus.CANCELLED:
                                job.updated = True

                        if gsheet_updater is not None:
                            logger.log("Pushing final status to sheets...")
                            # First update
                            dump_into_gsheet_queue(gsheet_updater, job_manager)
                            gsheet_updater.batch_update(force=True)
                            
                            # Double check all cancelled jobs are in the update queue
                            for job in job_manager.jobs:
                                if job.job_status == JobStatus.CANCELLED:
                                    gsheet_updater.add_to_queue(job.csv_row_id, MONITOR_STATUS_COLUMN, "CANCELLED")
                                    gsheet_updater.add_to_queue(
                                        job.csv_row_id,
                                        MONITOR_LAST_UPDATE_COLUMN,
                                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    )
                            
                            # Final update
                            logger.log("Waiting for sheet update to complete...")
                            gsheet_updater.batch_update(force=True)
                            time.sleep(2)  # Give time for update to propagate
                            logger.log("Sheet update completed.")

                except Exception as e:
                    logger.log(f"Failed to update sheets: {str(e)}")
                    logger.log("WARNING: Sheet updates may not have completed!")
                    time.sleep(5)  # Give a moment for the error to be visible
                    raise  # Re-raise to ensure we don't silently fail

                logger.log("All jobs cancelled and sheets updated.")
            except Exception as e:
                logger.log(f"Error during cancellation: {str(e)}")
                raise
            finally:
                # Only unset flags and exit after everything is truly done
                is_cancelling = False
                signal.signal(signal.SIGINT, original_sigint)
                signal.signal(signal.SIGTERM, original_sigterm)
                sys.exit(1)