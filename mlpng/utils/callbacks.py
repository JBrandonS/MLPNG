import sys
import tensorflow as tf

import numpy as np
import healpy as hp
from mlpng.utils import remove_mono_dipole


@tf.keras.saving.register_keras_serializable()
class RMSELoss(tf.keras.losses.Loss):
    """Custom RMSE loss function with serialization support."""

    def __init__(self, index=None, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index

    def call(self, y_true, y_pred):
        if self.index is None:
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        else:
            # Compute the loss for the specified index
            error = y_true[:, self.index] - y_pred[:, self.index]
            return tf.sqrt(tf.reduce_mean(tf.square(error)))

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


@tf.keras.saving.register_keras_serializable()
class RMSELoss2(tf.keras.losses.Loss):
    """Custom RMSE loss function with serialization support."""

    def __init__(self, index=None, name="rmse2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index

    def call(self, y_true, y_pred):
        if self.index is None:
            dom = 0.01 * y_true + 1e-6  # avoid zero division
            return tf.sqrt(tf.reduce_mean(tf.square((y_true - y_pred) / dom)))
        else:
            # Compute the loss for the specified index
            dom = 0.01 * y_true[:, self.index] + 1e-6  # avoid zero division
            error = (y_true[:, self.index] - y_pred[:, self.index]) / dom
            return tf.sqrt(tf.reduce_mean(tf.square(error)))

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


@tf.keras.saving.register_keras_serializable()
class RMSEMetric(tf.keras.metrics.Metric):
    """Custom RMSE metric with serialization support."""

    def __init__(self, index=None, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.index = index
        self.sum_squared_error = self.add_weight(name="sse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.index is None:
            error = y_true - y_pred
        else:
            error = y_true[:, self.index] - y_pred[:, self.index]

        squared_error = tf.square(error)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            squared_error = tf.multiply(squared_error, sample_weight)

        self.sum_squared_error.assign_add(tf.reduce_sum(squared_error))
        self.count.assign_add(tf.cast(tf.size(error), self.dtype))

    def result(self):
        return tf.sqrt(self.sum_squared_error / self.count)

    def reset_state(self):
        self.sum_squared_error.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


def rmse_metrics(shapes):
    """Generate a list of RMSE metrics for each shape."""
    metrics = []
    for i, shape in enumerate(shapes):
        metrics.append(RMSEMetric(index=i, name=f"rmse_{shape}"))
    return metrics


@tf.keras.saving.register_keras_serializable()
class PowerSpectrumLoss(tf.keras.losses.Loss):
    """Custom Power Spectrum loss function with serialization support."""

    def __init__(
        self,
        lmax,
        alpha=1.0,
        beta=1.0,
        use_pixel_weights=False,
        name="power_spectrum_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lmax = lmax
        self.alpha = alpha
        self.beta = beta
        self.use_pixel_weights = use_pixel_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lmax": self.lmax,
                "alpha": self.alpha,
                "beta": self.beta,
                "use_pixel_weights": self.use_pixel_weights,
            }
        )
        return config

    def _compute_cl_loss(self, y_t, y_p, use_pixel_weights):
        y_true_np = y_t.numpy()
        y_pred_np = y_p.numpy()

        losses = []  # probably want to remove the [] for empty list initialization
        for yt, yp in zip(y_true_np, y_pred_np):
            yt_map = hp.reorder(yt.T, n2r=True)
            yp_map = hp.reorder(yp.T, n2r=True)

            cl_true = hp.anafast(yt_map, use_pixel_weights=use_pixel_weights)
            cl_pred = hp.anafast(yp_map, use_pixel_weights=use_pixel_weights)

            cl_true = remove_mono_dipole(cl_true)
            cl_pred = remove_mono_dipole(cl_pred)

            losses.append(np.abs((cl_true - cl_pred) / cl_true))

        return np.mean(losses).astype(np.float32)

    @tf.function
    def call(self, y_true, y_pred):
        if self.alpha != 0.0:
            loss = tf.py_function(
                func=self._compute_cl_loss,
                inp=[y_true, y_pred, self.use_pixel_weights],
                Tout=tf.float32,
            )
            loss = tf.stop_gradient(loss)
        else:
            loss = tf.constant(0.0, dtype=tf.float32)

        # we need something to track gradients, so add a small pixel-wise loss
        pixel_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # tf.print(
        #     "Power Spectrum Loss:",
        #     loss,
        #     "Pixel Loss:",
        #     pixel_loss,
        #     output_stream=sys.stdout,
        # )

        return self.alpha * loss + self.beta * pixel_loss
