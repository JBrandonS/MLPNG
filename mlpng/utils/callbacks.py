import sys
import tensorflow as tf

import numpy as np
import healpy as hp
from mlpng.utils import remove_mono_dipole
from joblib import Parallel, delayed


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
            # Handle both concatenated output (single tensor) and separate heads (list of tensors)
            if isinstance(y_pred, list):
                # Multiple output heads: y_pred is a list of tensors
                pred_val = tf.squeeze(y_pred[self.index])
            else:
                # Single concatenated output: y_pred is a tensor, index into it
                pred_val = y_pred[:, self.index]

            true_val = y_true[:, self.index]
            error = true_val - pred_val
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
            # Handle both concatenated output (single tensor) and separate heads (list of tensors)
            if isinstance(y_pred, list):
                # Multiple output heads: y_pred is a list of tensors
                pred_val = tf.squeeze(y_pred[self.index])
            else:
                # Single concatenated output: y_pred is a tensor, index into it
                pred_val = y_pred[:, self.index]

            true_val = y_true[:, self.index]
            dom = 0.01 * true_val + 1e-6  # avoid zero division
            error = (true_val - pred_val) / dom
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
            # Handle both concatenated output (single tensor) and separate heads (list of tensors)
            if isinstance(y_pred, list):
                # Multiple output heads: y_pred is a list of tensors
                pred_val = tf.squeeze(y_pred[self.index])
            else:
                # Single concatenated output: y_pred is a tensor, index into it
                pred_val = y_pred[:, self.index]

            true_val = y_true[:, self.index]
            error = true_val - pred_val

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
        beta=1e-6,
        use_pixel_weights=True,
        name="power_spectrum_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.lmax = lmax
        self.alpha = alpha
        self.beta = beta
        self.use_pixel_weights = use_pixel_weights

        # TODO remove after testing
        self.iteration = tf.Variable(0, trainable=False, dtype=tf.int64)

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

    # def _compute_cl_loss(self, y_t, y_p, use_pixel_weights):
    #     y_true_np = y_t.numpy()
    #     y_pred_np = y_p.numpy()

    #     batch_size = y_true_np.shape[0]
    #     losses = np.zeros(batch_size, dtype=np.float32)

    #     # Reorder entire batch at once
    #     for i in range(batch_size):
    #         yt_map = hp.reorder(y_true_np[i].T, n2r=True)
    #         yp_map = hp.reorder(y_pred_np[i].T, n2r=True)

    #         yt_map = hp.remove_dipole(yt_map, copy=False)
    #         yp_map = hp.remove_dipole(yp_map, copy=False)

    #         cl_true = hp.anafast(yt_map, use_pixel_weights=use_pixel_weights, pol=False)
    #         cl_pred = hp.anafast(yp_map, use_pixel_weights=use_pixel_weights, pol=False)

    #         losses[i] = np.mean(np.square(cl_true - cl_pred))

    #     return np.mean(losses, dtype=np.float32)

    def _compute_cl_loss(self, y_t, y_p, use_pixel_weights):
        y_true_np = y_t.numpy()
        y_pred_np = y_p.numpy()

        def _per_example(i):
            yt_map = hp.reorder(y_true_np[i].T, n2r=True)
            yp_map = hp.reorder(y_pred_np[i].T, n2r=True)

            yt_map = hp.remove_monopole(yt_map, copy=False)
            yp_map = hp.remove_monopole(yp_map, copy=False)

            cl_true = hp.anafast(yt_map, use_pixel_weights=use_pixel_weights, pol=False)
            cl_pred = hp.anafast(yp_map, use_pixel_weights=use_pixel_weights, pol=False)

            return np.mean(np.square(cl_true - cl_pred))

        losses = Parallel(n_jobs=-1, backend="threading")(
            delayed(_per_example)(i) for i in range(y_true_np.shape[0])
        )
        return np.mean(losses, dtype=np.float32)

    @tf.function
    def call(self, y_true, y_pred):
        self.iteration.assign_add(1)

        if self.alpha != 0.0:
            loss = tf.py_function(
                func=self._compute_cl_loss,
                inp=[y_true, y_pred, self.use_pixel_weights],
                Tout=tf.float32,
            )
            loss = tf.stop_gradient(tf.reduce_mean(loss))
        else:
            loss = tf.constant(0.0, dtype=tf.float32)

        # we need something to track gradients, so add a small pixel-wise loss
        pixel_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # tf.cond(
        #     tf.equal(self.iteration % 63, 0),
        #     lambda: tf.print(
        #         " Step:",
        #         self.iteration,
        #         "PS Loss:",
        #         self.alpha * loss,
        #         "Pixel Loss:",
        #         self.beta * pixel_loss,
        #         output_stream=sys.stdout,
        #     ),
        #     lambda: tf.no_op(),  # no-op
        # )

        return self.alpha * loss + self.beta * pixel_loss


def _ps_loss_and_grad_numpy(y_true_np, y_pred_np, lmax):
    batch, npix, nch = y_pred_np.shape
    nside = hp.npix2nside(npix)

    losses = np.zeros(batch, dtype=np.float32)
    grads = np.zeros_like(y_pred_np, dtype=np.float32)

    for b in range(batch):
        yt_ring = hp.reorder(y_true_np[b, :, 0].numpy(), n2r=True)
        yp_ring = hp.reorder(y_pred_np[b, :, 0].numpy(), n2r=True)

        yt_ring = hp.remove_monopole(yt_ring, copy=False)
        yp_ring = hp.remove_monopole(yp_ring, copy=False)

        at = hp.map2alm(yt_ring, lmax=lmax)
        ap = hp.map2alm(yp_ring, lmax=lmax)

        cl_true = hp.alm2cl(at, lmax=lmax)
        cl_pred = hp.alm2cl(ap, lmax=lmax)

        losses[b] = np.mean(np.square(cl_true - cl_pred), dtype=np.float32)

        dloss_dcl = 2.0 * (cl_pred - cl_true) / cl_true.size / batch
        dalm = np.zeros_like(ap, dtype=np.complex128)

        for ell in range(lmax + 1):
            idx = hp.Alm.getidx(lmax, ell, np.arange(ell + 1))
            dalm[idx] = dloss_dcl[ell] * 2.0 * ap[idx] / (2 * ell + 1)

        grad_ring = hp.alm2map(dalm, nside=nside, lmax=lmax, pol=False)
        grads[b, :, 0] = hp.reorder(grad_ring.astype(np.float32), r2n=True)

    return np.mean(losses, dtype=np.float32), grads


@tf.custom_gradient
@tf.function
def _power_spectrum_cl_loss(y_true, y_pred, lmax):
    loss, grads = tf.py_function(
        _ps_loss_and_grad_numpy,
        [y_true, y_pred, lmax],
        [tf.float32, tf.float32],
    )
    loss.set_shape([])
    grads.set_shape(y_pred.shape)

    def backward(dy):
        return tf.zeros_like(y_true), dy * grads, None, None

    return loss, backward


@tf.keras.saving.register_keras_serializable()
class PowerSpectrumLoss2(tf.keras.losses.Loss):
    def __init__(self, lmax, name="power_spectrum_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lmax = lmax

    def get_config(self):
        config = super().get_config()
        config.update({"lmax": self.lmax})
        return config

    @tf.function
    def call(self, y_true, y_pred):
        return _power_spectrum_cl_loss(y_true, y_pred, self.lmax)
