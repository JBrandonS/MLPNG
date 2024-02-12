import time

import tensorflow as tf


class TimedLoggingCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback for logging training progress and time.

    This callback logs the progress of training along with the time taken for each batch and epoch.
    The frequency of logging can be controlled with the `print_frequency` parameter.

    Attributes:
        print_frequency (int): The frequency of logging in seconds. Default is 60 seconds.
        last_print_time (float): The last time the log was printed.
        batch_start_time (float): The start time of the current batch.
        epoch_start_time (float): The start time of the current epoch.
        first_epoch (bool): A flag to check if it's the first epoch.
    """
        
    def __init__(self, print_frequency=60):
        super().__init__()
        self.print_frequency = print_frequency
        self.last_print_time = time.time()
        self.batch_start_time = 0
        self.epoch_start_time = 0
        self.first_epoch = True

    def _get_time_str(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        if minutes > 60:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02}:{seconds:02}"
        return f"{minutes}:{seconds:02}" if minutes else f"{seconds}s"

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_frequency:
            eta = (self.params["steps"] - batch) * (
                current_time - self.batch_start_time
            )
            eta = self._get_time_str(eta)

            progress = batch / self.params["steps"]
            progress_bar = (
                "=" * int(progress * 30) + ">" + "." * (29 - int(progress * 30))
            )

            metrics_log = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())

            batch_str = str(batch).rjust(len(str(self.params["steps"])))
            print(
                f"{batch_str}/{self.params['steps']} [{progress_bar}] - ETA: {eta} - {metrics_log}",
                flush=True,
            )
            self.last_print_time = current_time

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

        if self.first_epoch:
            print("Starting training...", flush=True)
            self.first_epoch = False

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        
        elapsed_time = time.time() - self.epoch_start_time
        elapsed_time = self._get_time_str(elapsed_time)

        metrics_log = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())

        epoch_str = str(epoch).rjust(len(str(self.params["epochs"])))
        print(f"Epoch: {epoch_str} - Time: {elapsed_time} - {metrics_log}", flush=True)
        self.last_print_time = time.time()
