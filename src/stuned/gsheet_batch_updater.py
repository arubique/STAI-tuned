import copy
import time
from collections import defaultdict

import gspread

from stuned.utility.logger import log_csv_for_concurrent, read_csv_as_dict_lock
from stuned.utility.utils import dicts_not_equal, retrier_factory, format_free_equal


# from stuned.run_from_csv.__main__ import MONITOR_LAST_UPDATE_COLUMN


class GSheetBatchUpdater:
    def __init__(
        self,
        spreadsheet_url,
        worksheet_name,
        gsheet_client: gspread.client,
        logger,
        csv_path,
        input_csv,
        update_interval=10,
    ):
        self.spreadsheet_url = spreadsheet_url
        self.worksheet_name = worksheet_name
        self.gsheet_client = gsheet_client
        self.logger = logger

        # Stores the updates in a queue as a nested dictionary: {row_id: {col: value}}
        self.queue = defaultdict(dict)
        self.update_interval = update_interval
        self.last_update = -1
        self.csv_path = csv_path
        # keep an internal flag to check if the last update was successful. If it was not, we can try again and skipping
        # the queue.
        self.last_update_status = False

        # We also get the local csv. we use it for periodically checking for updates in csv so that we can report them globally
        # instead of relying on individual processes to report the changes. Note that we need to take only the rows that acutally changed
        self.local_csv = input_csv

        # order-preserving list of column names
        self.col_names = list(self.local_csv[0])

    def add_to_queue(self, row, col, value):
        """Add an update to the queue."""
        self.queue[row][
            col
        ] = value  # This will set the value for the specific column in the specific row

        # if column isn't in the local csv, add it and mark all other rows as empty value there
        if col not in self.col_names:
            self.col_names.append(col)
            for row_id in self.local_csv:
                if col not in self.local_csv[row_id]:
                    self.local_csv[row_id][col] = ""

    def batch_update(self, force=False):
        """Batch update the Google Sheet with the queued changes."""
        if len(self.queue.keys()) == 0 or (
            not force
            and (self.last_update_status and time.time() - self.last_update < self.update_interval)
        ):
            return True
        # Convert the queue to the report_stuff_ids format
        # report_stuff_ids = [(row, col, value) for row, col_value_dict in self.queue.items() for col, value in
        #                     col_value_dict.items()]
        # Rewrite the above in a full loop
        report_csv_updates = []
        affected_rows = set()
        for row, col_value_dict in self.queue.items():
            for col, value in col_value_dict.items():
                if (
                    self.local_csv is not None
                    and row in self.local_csv
                    and col in self.local_csv[row]
                    and (format_free_equal(self.local_csv[row][col], value))
                ):
                    continue
                report_csv_updates.append((row, col, value))
                affected_rows.add(row)
                # update local csv
                self.local_csv[row][col] = value
        affected_rows = list(affected_rows)
        
        try:
            # TODO: only update if there are changes in columns

            self.gsheet_client.upload_csvs_to_spreadsheet_no_csv(
                self.local_csv,
                self.spreadsheet_url,
                self.worksheet_name,
                affected_rows=[0],
            )
            # update all other stuff
            if len(affected_rows) > 0:
                self.gsheet_client.upload_csvs_to_spreadsheet_no_csv(
                    self.local_csv,
                    self.spreadsheet_url,
                    self.worksheet_name,
                    affected_rows=affected_rows,
                )
            # New way: don't rely on csvs: use dict directly

        except Exception as e:
            self.logger.log(f"Exception while uploading to GSheet: {e}")
            self.logger.log("Will repeat later...")
            self.last_update_status = False
            return False
        self.queue.clear()  # Clear the queue after updating

        self.last_update = time.time()  # Update the last update time
        self.last_update_status = True
        return True

    def update_remote_if_changes_happened(self):
        """Iterates through the current CSV and compares it with the local CSV. If there are changes, update the remote sheet"""

        csv_current = read_csv_as_dict_lock(self.csv_path)

        affected_rows = []
        if self.local_csv is None:
            affected_rows = list(csv_current.keys())
        else:
            for row_id, row in csv_current.items():
                if row_id not in self.local_csv or dicts_not_equal(row, self.local_csv[row_id]):
                    affected_rows.append(row_id)

        # TODO: refactor this since this exact code is also used in `batch_update``
        try:
            self.gsheet_client.upload_csvs_to_spreadsheet(
                self.spreadsheet_url,
                [self.csv_path],
                [self.worksheet_name],
                single_rows_per_csv=[[0] + affected_rows],
            )
        except Exception as e:
            self.logger.log(f"Exception while uploading to GSheet: {e}")
            self.logger.log("Will repeat later?")
            self.last_update_status = False
            return False

        self.local_csv = copy.deepcopy(csv_current)

        return True
