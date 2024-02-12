import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np

import h5py


@tf.function(jit_compile=True)
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


@tf.function(jit_compile=True)
def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)