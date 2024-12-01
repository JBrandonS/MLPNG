import logging

import numpy as np
import keras
from keras.optimizers.schedules import LearningRateSchedule

logger = logging.getLogger(__name__)


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
        warmup_learning_rate=1e-3,
        warmup_steps=10000,
        warmup_scale=1.5,
        warmup_scale_steps=10,
        warmed_learning_rate="auto",
        decay_steps=10000,
        decay_rate=0.95,
        staircase=True,
        ftype=np.float32,
        itype=np.int32,
    ):
        logger = logging.getLogger(self.__class__.__name__)

        self.ftype = ftype
        self.itype = itype

        # casting everything because tf was yelling about it
        self.warmup_learning_rate = keras.ops.cast(warmup_learning_rate, ftype)
        self.warmup_steps = keras.ops.cast(warmup_steps, itype)
        self.warmup_scale = keras.ops.cast(warmup_scale, ftype)
        self.warmup_scale_steps = keras.ops.cast(warmup_scale_steps, itype)

        max_warmup = warmup_learning_rate * (
            1 + warmup_scale * (warmup_steps / warmup_scale_steps)
        )
        logger.info("Warmup Range: %s -> %s", warmup_learning_rate, max_warmup)

        # these should follow the ExponentialDecay function
        if warmed_learning_rate == "auto":
            self.warmed_learning_rate = keras.ops.cast(max_warmup, ftype)
        else:
            self.warmed_learning_rate = keras.ops.cast(warmed_learning_rate, ftype)
        self.decay_steps = keras.ops.cast(decay_steps, itype)
        self.decay_rate = keras.ops.cast(decay_rate, ftype)
        self.staircase = staircase

    # @tf.function
    def __call__(self, step):
        step = keras.ops.cast(step, self.itype)
        if step < self.warmup_steps:
            p_warmup = keras.ops.cast(step / self.warmup_scale_steps, self.ftype)
            if self.staircase:
                p_warmup = tf.floor(p_warmup)
            scale_warmup = tf.multiply(self.warmup_scale, p_warmup)
            return tf.multiply(self.warmup_learning_rate, 1.0 + scale_warmup)
        else:
            p_decay = keras.ops.cast(
                (step - self.warmup_steps) / self.decay_steps, self.ftype
            )
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


class AttentionSchedule(LearningRateSchedule):
    """Taken from the attention is all you need paper

    See: https://www.tensorflow.org/text/tutorials/transformer
    """

    def __init__(self, d_model, warmup_steps=4000):
        # super().__init__()
        self.d_model = d_model
        self.d_model = keras.ops.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = keras.ops.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
