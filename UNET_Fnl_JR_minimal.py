#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/hpc/mp/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-10.3.0/cuda-11.4.4-ctldo35wmmwws3jbgwkgjjcjawddu3qz/"

import tensorflow as tf
from tensorflow.keras import backend as K
import sklearn.model_selection as model_selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tf.config.list_physical_devices('GPU')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from functools import partial

from tensorflow.keras import Input
from tensorflow.keras.layers import Attention, Layer, LeakyReLU, Add, UpSampling2D, Activation, SpatialDropout2D, Conv2D, BatchNormalization, Concatenate, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization

from tensorflow import pad


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        # self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def create_localization_module(input_layer, n_filters):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2)):
    up_sample = UpSampling2D(size=size)(input_layer)
    layer1 = ReflectionPadding2D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    layer1 = ReflectionPadding2D()(input_layer)
    convolution1 = create_convolution_block(input_layer=layer1, n_filters=n_level_filters)
    dropout = SpatialDropout2D(rate=dropout_rate, data_format=data_format)(convolution1)
    layer2 = ReflectionPadding2D()(dropout)
    convolution2 = create_convolution_block(input_layer=layer2, n_filters=n_level_filters)
    return convolution2


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3), activation=None,
                             padding='valid', strides=(1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = InstanceNormalization()(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

def isensee2017_model(inputs, n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=dice_coefficient_loss, activation_name="relu"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param inputs:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
#         n_level_filters = (2**level_number) * n_base_filters
        n_level_filters = 16
        #n_level_filters = 32
        #n_level_filters = 64
        #n_level_filters = 48
        level_filters.append(n_level_filters)
        

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters, strides=(2, 2))
            #in_conv = create_convolution_block(layer, n_level_filters, strides=(3, 3))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        
        #attention_layer = Attention(use_scale=True, dropout=dropout_rate)([level_output_layers[level_number], up_sampling])
        
        #concatenation_layer = Concatenate()([level_output_layers[level_number], up_sampling, attention_layer])
        
        concatenation_layer = Concatenate()([level_output_layers[level_number], up_sampling])
        
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        
        current_layer = localization_output
        
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels-1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2))(output_layer)
            #output_layer = UpSampling2D(size=(3, 3))(output_layer)

    flat_layer = Flatten()(output_layer)
    
    out_layer = Dense(1,activation=None)(flat_layer)
    
    # model = Model(inputs=inputs, outputs=activation_block)
    model = Model(inputs=inputs, outputs=out_layer)
    model.compile(optimizer=optimizer(learning_rate=initial_learning_rate), loss=loss_function, metrics=tf.keras.metrics.RootMeanSquaredError())
    return model

xp = 128
yp = 128
XX_0 = np.load(f"/work/users/jwryan/NGM/NGTmap_sampled_fNL_1000_range_128x128_ngmap_512_10k_batch_1_0_tvt_uncontaminated.npz")

NUMBATCH = 10
if NUMBATCH > 1:
    BATCHSTR = 'batches'
if NUMBATCH == 1:
    BATCHSTR = 'batch'

k_list = []
for j in range(1, NUMBATCH + 1, 1): #1, 11, 1
        print(j)
        XX_list = []
        yy_list = []

        kcount = 0
        for i in range(0, 1000, 1): #0, 1000, 1
            #print(i, 'Batch', j)
            
            XX_data_test_i = np.load(f"/work/users/jwryan/NGM/NGTmap_sampled_fNL_1000_range_128x128_ngmap_512_10k_batch_" + str(j) + "_" + str(i) + "_tvt_uncontaminated.npz")
            XX_data_test_ii = XX_data_test_i['Tmapdata'][:,:,:,np.newaxis]
            yy_data_test_ii = XX_data_test_i['fNLs'][:,0]
            
            XX_list.append(XX_data_test_ii)
            yy_list.append(yy_data_test_ii)
            
        if j == 1:
            XX = np.vstack(XX_list)
            yy = np.ndarray.flatten(np.vstack(yy_list))

        if j > 1:
            XX_1 = np.vstack(XX_list)
            yy_1 = np.ndarray.flatten(np.vstack(yy_list))

            XX = np.concatenate((XX, XX_1))
            yy = np.concatenate((yy, yy_1))

learning_rate = 0.001 #0.001, 0.0001, 0.00001
Batch_size = 32 #32, 100, 16
num_epochs = 10
drop_rate = 0.3 #0.5, 0.3, 0.7, 0.9, 0.99 (originally 0.3)
halt_patience = 6 #6, 10, 20 (originally 6). Most recent setting, as of 2023 Dec 12: 10.
lr_decay = 0.5
lrd_patience = 3 #10, 3, 20 (originally 3). Most recent setting, as of 2023 Dec 12: 10.

save_model = False
#save_model = True

# creating callbacks for training
halt = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=halt_patience, restore_best_weights=True)
halvelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=lr_decay, patience=lrd_patience)
csv_logger = tf.keras.callbacks.CSVLogger('training.log', separator=',', append=False)

callbacks = [halt, halvelr, csv_logger]

K.clear_session()

input_img = Input((128, 128, 1), name='img') #original setting
#input_img = Input((256, 256, 1), name='img')
DEPTH = 5 #7
model = isensee2017_model(input_img, depth=5, n_segmentation_levels=3, dropout_rate=drop_rate, loss_function=tf.keras.losses.mse, initial_learning_rate=learning_rate)
model.summary()

K.set_value(model.optimizer.learning_rate, learning_rate)
print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())

BEGIN = 0
END = 1
epoch_count = 10*BEGIN

train_loss_list = []
val_loss_list = []

for i in range(BEGIN, END, 1):
    if i == 0:
        train_split, val_split, test_split = 0.8, 0.1, 0.1
        random_seed = 42
        
        def split_data(dsetx, dsety, tr=0.8, va=0.1, te=0.1, rseed=42):
            val_remain = val_split / (1 - test_split)
            X_remain, X_test, y_remain, y_test = model_selection.train_test_split(
                                                    dsetx, dsety, test_size=te, random_state=rseed)
        
            X_train, X_val, y_train, y_val = model_selection.train_test_split(
                                                    X_remain, y_remain, test_size=val_remain, random_state=rseed+1)
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(XX, yy)
        print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
        
        np.savez('X_train_no_attention_tvt_uncontaminated.npz', key=X_train)
        np.savez('X_val_no_attention_tvt_uncontaminated.npz', key=X_val)
        np.savez('X_test_no_attention_tvt_uncontaminated.npz', key=X_test)
        np.savez('y_train_no_attention_tvt_uncontaminated.npz', key=y_train)
        np.savez('y_val_no_attention_tvt_uncontaminated.npz', key=y_val)
        np.savez('y_test_no_attention_tvt_uncontaminated.npz', key=y_test)

    if i > 0:
        model.load_weights('/users/jwryan/NN-pnG-main/NN-pnG-main/10k_' + str(NUMBATCH) + '_' + BATCHSTR + '/single_run_10k_20_batches_n_level_filters_16_depth_5_checkpoint_' + str(epoch_count) + '_epochs_no_attention_tvt_uncontaminated')
        
        X_train = np.load('X_train_no_attention_tvt_uncontaminated.npz')
        X_train = X_train['key']
        X_val = np.load('X_val_no_attention_tvt_uncontaminated.npz')
        X_val = X_val['key']
        X_test = np.load('X_test_no_attention_tvt_uncontaminated.npz')
        X_test = X_test['key']
        y_train = np.load('y_train_no_attention_tvt_uncontaminated.npz')
        y_train = y_train['key']
        y_val = np.load('y_val_no_attention_tvt_uncontaminated.npz')
        y_val = y_val['key']
        y_test = np.load('y_test_no_attention_tvt_uncontaminated.npz')
        y_test = y_test['key']
        
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=Batch_size, epochs=num_epochs, callbacks=callbacks)
    
    epoch_count += 10
    
    model.save_weights('/users/jwryan/NN-pnG-main/NN-pnG-main/10k_' + str(NUMBATCH) + '_' + BATCHSTR + '/single_run_10k_20_batches_n_level_filters_16_depth_5_checkpoint_' + str(epoch_count) + '_epochs_no_attention_tvt_uncontaminated')
    
    if save_model:
        model.save_weights(
            'model_weights_valloss-{0:.4f}.h5').format(min(history.history['val_loss']))
    
        with open('model_architecture.json', 'w') as f:
            f.write(model.to_json())

