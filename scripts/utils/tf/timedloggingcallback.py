import time
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from keras.utils.io_utils import print_msg

class TimedLoggingCallback(Callback):
    """
    A custom Keras callback for logging training progress and time.

    This callback logs the progress of training along with the time taken for each batch and epoch.
    The frequency of logging can be controlled with the `print_frequency` parameter.

    Attributes:
        print_frequency (int): The frequency of logging in seconds. Default is 60 seconds.
        last_print_time (float): The last time the log was printed.
        batch_start_time (float): The start time of the current batch.
        epoch_start_time (float): The start time of the current epoch.
    """

    def __init__(self, print_frequency=60):
        super().__init__()

        self.print_frequency = print_frequency
        self.last_print_time = 0
        self.batch_start_time = 0
        self.epoch_start_time = 0

    def set_params(self, params):
        super().set_params(params)

        # some small precalcs because tf was yelling about time
        self._epoch_str_len = len(str(self.params["epochs"]))
        self._steps_str_len = len(str(self.params["steps"]))

    @staticmethod
    def _get_time_str(seconds):
        """
        Convert seconds to a human readable string.

        Examples:
            _get_time_str(5546) -> "1:32:26"
            _get_time_str(1946) -> "32:26"
            _get_time_str(26)   -> "26s"
        """
        seconds = int(seconds)
        if seconds >= 60:
            minutes, seconds = divmod(seconds, 60)
            if minutes >= 60:
                hours, minutes = divmod(minutes, 60)
                return f"{hours}:{minutes:02}:{seconds:02}"
            return f"{minutes}:{seconds:02}"
        return f"{seconds:02}s"

    @staticmethod
    def _get_log_line(logs=None):
        """
        Convert the logs to a human readable string of format ' - key: value'.
        """
        if logs is None:
            return ""
        return " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())

    def on_train_begin(self, logs=None):
        print_msg("Starting training...", flush=True)
        self.last_print_time = time.time()

    def on_train_end(self, logs=None):
        print_msg("Training complete.")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_frequency:
            steps = self.params["steps"]

            eta = (steps - batch) * (current_time - self.batch_start_time)
            eta = self._get_time_str(eta)

            progress = batch / steps
            progress_bar_val = int(progress * 30)
            progress_bar = "=" * progress_bar_val + ">" + "." * (29 - progress_bar_val)

            metrics_log = self._get_log_line(logs)

            batch_str = f"{str(batch).rjust(self._steps_str_len)} / {steps}"
            print_msg(f"{batch_str} [{progress_bar}] - ETA: {eta} - {metrics_log}")
            self.last_print_time = current_time

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        elapsed_time = self._get_time_str(elapsed_time)

        metrics_log = self._get_log_line(logs)

        epoch_str = str(epoch).rjust(self._epoch_str_len)

        print_msg(
            f"Epoch: {epoch_str} / {self.params['epochs']} - Time: {elapsed_time} - {metrics_log}"
        )
        self.last_print_time = current_time
