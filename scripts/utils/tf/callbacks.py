import time
import logging

import tensorflow as tf
from tensorflow.config import list_physical_devices
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class WarmupLearningRate(LearningRateSchedule):
    """
    A learning rate schedule that starts with a warm-up period where the learning rate increases,
    followed by a period where the learning rate decays.

    Attributes:
        warmup_learning_rate (float): The initial learning rate during the warm-up period.
        warmup_steps (int): The number of steps over which the learning rate increases.
        warmup_scale (float): The scale factor for the learning rate increase during the warm-up period.
        warmup_scale_steps (int): The number of steps over which the warm-up scale is applied.
        warmed_learning_rate (float): The learning rate after the warm-up period.
        decay_steps (int): The number of steps over which the learning rate decays.
        decay_rate (float): The decay rate for the learning rate after the warm-up period.
        staircase (bool): If True, the learning rate changes in discrete steps, otherwise, it changes smoothly.

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
        warmed_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True,
        ftype=tf.float32,
        itype=tf.int32,
    ):
        self.logger = logging.getLogger("WarmupLearningRate")
        self.logger.setLevel(logging.INFO)

        self.ftype = ftype
        self.itype = itype

        # casting everything because tf was yelling about it
        self.warmup_learning_rate = tf.cast(warmup_learning_rate, ftype)
        self.warmup_steps = tf.cast(warmup_steps, itype)
        self.warmup_scale = tf.cast(warmup_scale, ftype)
        self.warmup_scale_steps = tf.cast(warmup_scale_steps, itype)

        max_warmup = warmup_learning_rate * (
            1 + warmup_scale * (warmup_steps / warmup_scale_steps)
        )
        self.logger.info(f"Warmup Range: {warmup_learning_rate} -> {max_warmup}")

        # these should follow the ExponentialDecay function
        if warmed_learning_rate == "auto":
            self.warmed_learning_rate = tf.cast(max_warmup, ftype)
        else:
            self.warmed_learning_rate = tf.cast(warmed_learning_rate, ftype)
        self.decay_steps = tf.cast(decay_steps, itype)
        self.decay_rate = tf.cast(decay_rate, ftype)
        self.staircase = staircase

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, self.itype)
        if step < self.warmup_steps:
            p_warmup = tf.cast(step / self.warmup_scale_steps, self.ftype)
            if self.staircase:
                p_warmup = tf.floor(p_warmup)
            scale_warmup = tf.multiply(self.warmup_scale, p_warmup)
            return tf.multiply(self.warmup_learning_rate, 1.0 + scale_warmup)
        else:
            p_decay = tf.cast((step - self.warmup_steps) / self.decay_steps, self.ftype)
            if self.staircase:
                p_decay = tf.floor(p_decay)
            scale_decay = tf.pow(self.decay_rate, p_decay)
            return tf.multiply(self.warmed_learning_rate, scale_decay)

    def get_config(self):
        return {
            "warmup_learning_rate": self.warmup_learning_rate.numpy(),  # type: ignore
            "warmup_steps": self.warmup_steps.numpy(),  # type: ignore
            "warmup_scale": self.warmup_scale.numpy(),  # type: ignore
            "warmup_scale_steps": self.warmup_scale_steps.numpy(),  # type: ignore
            "warmed_learning_rate": self.warmed_learning_rate.numpy(),  # type: ignore
            "decay_steps": self.decay_steps.numpy(),  # type: ignore
            "decay_rate": self.decay_rate.numpy(),  # type: ignore
            "staircase": self.staircase,
            "ftype": self.ftype.name,
            "itype": self.itype.name,
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
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}:{minutes:02}:{seconds:02}"
        elif minutes > 0:
            return f"{minutes}:{seconds:02}"
        else:
            return f"{seconds:>2}s"

    def _get_log_line(self, logs=None):
        if logs is None:
            return ""
        return " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())

    def _lr_str(self):
        return f"Learning Rate: {self.model.optimizer.lr.numpy().item():.4f}"

    def on_train_begin(self, logs=None):
        print("Starting training...")
        self.last_print_time = time.time()
        self.num_replicas = len(list_physical_devices("GPU")) or 1

    def on_train_end(self, logs=None):
        print("Training complete.")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_frequency:
            steps = self.params["steps"]

            batch_str = f"{str(batch).rjust(self._steps_str_len)}/{steps}"

            progress = batch / steps
            progress_bar_val = int(progress * 30)
            progress_bar = "=" * progress_bar_val + ">" + "." * (29 - progress_bar_val)

            metrics_log = self._get_log_line(logs)

            eta = (
                (steps - batch)
                * (current_time - self.batch_start_time)
                / self.num_replicas
            )
            eta = self._get_time_str(eta)

            print(
                f"\r{batch_str} [{progress_bar}] - ETA: {eta} - {metrics_log} - {self._lr_str()}",
                end=''
            )
            self.last_print_time = current_time

    def on_epoch_begin(self, epoch, logs=None):
        self.last_print_time = self.epoch_start_time = time.time()
        # print(f"Starting epoch {epoch + 1}/{self.params['epochs']}...")

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        epoch_str = f"{str(epoch).rjust(self._epoch_str_len)}/{self.params['epochs']}"
        elapsed_time = self._get_time_str(current_time - self.epoch_start_time)
        metrics_log = self._get_log_line(logs)
        print(
            f"\rEpoch: {epoch_str} - Time: {elapsed_time} - {metrics_log} - {self._lr_str()}"
        )
