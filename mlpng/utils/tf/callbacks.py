import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class BurnInLearningRate(LearningRateSchedule):
    """I wanted to start with a very high lr and then drop it down to a lower one"""

    # might be good to convert this to full class and TF like the lr_schedule
    def __init__(self, initial_learning_rate):
        self.burnin_learning_rate = 0.01  # specify initial learning rate
        self.burnin_epochs = 1  # specify the number of epochs for the burnin

        # these should follow the ExponentialDecay function
        self.learning_rate = initial_learning_rate  # specify secondary learning rate
        self.decay_steps = 5  # specify the number of epochs for decaying
        self.decay_rate = 0.96  # specify the decay rate
        self.staircase = True

    @tf.function
    def __call__(self, step):
        p = (step - self.burnin_epochs) / (self.decay_steps)
        if self.staircase:
            p = tf.floor(p)
        p = tf.cast(p, tf.float32)

        return tf.cond(
            step < self.burnin_epochs,
            lambda: self.burnin_learning_rate,
            lambda: tf.multiply(self.learning_rate, tf.pow(self.decay_rate, p)),
        )

    def get_config(self):
        return {
            "burnin_learning_rate": self.burnin_learning_rate,
            "burnin_epochs": self.burnin_epochs,
            "learning_rate": self.learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
        }


class WarmupLearningRate(LearningRateSchedule):
    """
    A learning rate schedule that starts with a warm-up period where the learning rate increases,
    followed by a period where the learning rate decays.

    Attributes:
        warmup_learning_rate (float): The initial learning rate during the warm-up period.
        warmup_steps (int): The number of steps over which the learning rate increases.
        warmup_scale (float): The scale factor for the learning rate increase during the warm-up period.
        warmup_scale_steps (int): The number of steps over which the warm-up scale is applied.
        base_learning_rate (float): The base learning rate after the warm-up period.
        decay_steps (int): The number of steps over which the learning rate decays.
        decay_rate (float): The decay rate for the learning rate after the warm-up period.
        staircase (bool): If True, the learning rate changes in discrete steps, otherwise, it changes smoothly.
        dtype (tf.DType): The data type for the learning rate and steps.

    Methods:
        __call__(step): Returns the learning rate for the given step.
        get_config(): Returns a dictionary containing the configuration of the learning rate schedule.
    """

    def __init__(
        self,
        warmup_learning_rate,
        warmup_steps,
        warmup_scale,
        warmup_scale_steps,
        base_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True,
        dtype=tf.float32,
    ):
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = int(warmup_steps)
        self.warmup_scale = warmup_scale
        self.warmup_scale_steps = warmup_scale_steps

        # these should follow the ExponentialDecay function
        self.base_learning_rate = base_learning_rate
        self.decay_steps = int(decay_steps)
        self.decay_rate = decay_rate
        self.staircase = staircase

        self.dtype = dtype

    @tf.function
    def __call__(self, step):
        def warmup_fn():
            """Applies warm-up to the learning rate."""
            p = 1 + step / self.warmup_scale_steps
            p = tf.cast(p, self.dtype)
            if self.staircase:
                p = tf.floor(p)
            scale = tf.multiply(self.warmup_scale, p)
            return tf.multiply(self.warmup_learning_rate, 1 + scale)

        def decay_fn():
            """Applies exponential decay to the learning rate."""
            p = (step - self.warmup_steps) / (self.decay_steps)
            p = tf.cast(p, self.dtype)
            if self.staircase:
                p = tf.floor(p)
            scale = tf.pow(self.decay_rate, p)
            return tf.multiply(self.base_learning_rate, scale)
        
        # tf.print("LR Rate:", step, "<", self.warmup_steps, "(", warmup_fn(), " ", decay_fn(), ")")
        return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)

    def get_config(self):
        return {
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
            "warmup_scale": self.warmup_scale,
            "warmup_scale_steps": self.warmup_scale_steps,
            "base_learning_rate": self.base_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "dtype": self.dtype.name,
        }


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

    def _get_time_str(self, seconds):
        """
        Convert seconds to a human readable time string.
        Tracks the depth of the time string per epoch to ensure consistent formatting.
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours:>2}:{minutes:02}:{seconds:02}"
        elif minutes > 0:
            return f"{minutes:>2}:{seconds:02}"
        else:
            return f"{seconds:>2}s"

    @staticmethod
    def _get_log_line(logs=None):
        if logs is None:
            return ""
        return " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())

    def on_train_begin(self, logs=None):
        tf.print("Starting training...")
        self.last_print_time = time.time()
        self.num_replicas = (
            len(tf.config.experimental.list_physical_devices("GPU")) or 1
        )

    def on_train_end(self, logs=None):
        tf.print("Training complete.")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0: # skip the first batch
            return
        
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_frequency:
            steps = self.params["steps"]

            eta = (
                (steps - batch)
                * (current_time - self.batch_start_time)
                / self.num_replicas
            )
            eta = self._get_time_str(eta)

            progress = batch / steps
            progress_bar_val = int(progress * 30)
            progress_bar = "=" * progress_bar_val + ">" + "." * (29 - progress_bar_val)

            metrics_log = self._get_log_line(logs)

            batch_str = f"{str(batch).rjust(self._steps_str_len)}/{steps}"

            tf.print(f"{batch_str} [{progress_bar}] - ETA: {eta} - {metrics_log}")
            self.last_print_time = current_time

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.last_print_time = current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        elapsed_time = self._get_time_str(elapsed_time)
        metrics_log = self._get_log_line(logs)
        epoch_str = f"{str(epoch).rjust(self._epoch_str_len)}/{self.params['epochs']}"
        tf.print(f"Epoch: {epoch_str} - Time: {elapsed_time} - {metrics_log}")
