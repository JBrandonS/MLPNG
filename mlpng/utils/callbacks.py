import tensorflow as tf


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
