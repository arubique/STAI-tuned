from typing import Dict, List

from stuned.utility.message_client import MessageType


# Create enum of job status
class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    slurm_submitted_statuses = ["PENDING", "CONFIGURING"]
    slurm_running_statuses = ["RUNNING", "COMPLETING"]
    slurm_failed_statuses = ["FAILED"]
    slurm_timeout_statuses = ["TIMEOUT"]
    slurm_cancelled_statuses = ["CANCELLED", "CANCELLED+"]
    slurm_completed_statuses = ["COMPLETED", "COMPLETED+"]

def get_slurm_job_status(slurm_status: str) -> str:
    if slurm_status in JobStatus.slurm_running_statuses:
        job_status = JobStatus.RUNNING
    elif slurm_status in JobStatus.slurm_completed_statuses:
        job_status = JobStatus.COMPLETED
    elif slurm_status in JobStatus.slurm_submitted_statuses:
        job_status = JobStatus.PENDING
    elif slurm_status in JobStatus.slurm_failed_statuses:
        job_status = JobStatus.FAILED
    elif slurm_status in JobStatus.slurm_cancelled_statuses:
        job_status = JobStatus.CANCELLED
    elif slurm_status in JobStatus.slurm_timeout_statuses:
        job_status = JobStatus.TIMEOUT
    else:
        job_status = JobStatus.UNKNOWN + "_" + (slurm_status or "None")
    return job_status


class Job:
    def __init__(self, job_id, job_status, job_exit_code, csv_row_id):
        self.job_id = job_id
        self.job_status = job_status
        self.job_exit_code = job_exit_code
        self.csv_row_id = csv_row_id

        self.updated = True

        self.writing_queue = []

    def set_job_id(self, job_id):
        self.job_id = job_id

    def set_job_status(self, job_status):
        self.job_status = job_status

    def set_exit_code(self, job_exit_code):
        self.job_exit_code = job_exit_code

    def process_message(self, message: List[Dict]):
        for message in message["messages"]:
            assert "type" in message, "Message type not found"
            assert "key" in message, "Message key not found"
            assert "value" in message, "Message value not found"

            msg_type, msg_key, msg_value = message["type"], message["key"], message["value"]

            if msg_type == MessageType.JOB_STARTED:
                # Simply change the status of the job
                self.set_job_status(JobStatus.RUNNING)
            elif msg_type == MessageType.JOB_RESULT_UPDATE:
                # Write to queue
                self.writing_queue.append((msg_key, msg_value))
            self.updated = True


def find_job_idx(jobs: List[Job], job_id: int) -> int:
    for idx, job in enumerate(jobs):
        if job.job_id == job_id:
            return idx
    return None


def find_job_id_by_row_id(jobs: List[Job], row_id: int) -> Job:
    for idx, job in enumerate(jobs):
        if job.csv_row_id == row_id:
            return idx
    return None
