import tensorflow as tf
from keras.layers import (
    Add,
    Attention,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    Multiply,
    UpSampling2D,
    RandomFlip,
    RandomRotation,
)
from keras.models import Model
from keras.optimizers.legacy import Adam

from tf_utils import (
    dice_coefficient_loss,
    ReflectionPadding2D,
    create_context_module,
    create_convolution_block,
    create_up_sampling_module,
    create_localization_module,
    rotation_layer
)


def isensee_attn(
    inputs,
    n_base_filters=16,
    depth=5,
    dropout_rate=0.3,
    n_segmentation_levels=3,
    n_labels=8,
    optimizer=Adam,
    initial_learning_rate=5e-4,
    loss_function=dice_coefficient_loss,
    name="",
    metrics=[],
    interpolation="nearest",
    kernel_regularizer=None,
    attn_heads=2,
    attn_key_dim=64,
    flip=True,
    rotate=False,
    add_t2=False,
):
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

    if flip:
        # flips vertical and horizontal
        current_layer = RandomFlip()(current_layer)

    if rotate:
        # random rotation
        current_layer = rotation_layer(current_layer)

    if add_t2:
        # Squares every pixel and add them, gets T2 map
        squared = Lambda(lambda x: tf.square(x), name="T_squared")(current_layer)
        current_layer = Concatenate()([current_layer, squared])

    level_output_layers = []
    level_filters = []
    for level in range(depth):
        n_level_filters = n_base_filters // (2**level)
        n_level_filters = max(4, n_level_filters)
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding2D()(current_layer)
            in_conv = create_convolution_block(
                layer,
                n_level_filters,
                strides=(2, 2),
                kernel_regularizer=kernel_regularizer,
            )

        # Self-attention
        # attention = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(
        #     in_conv, in_conv
        # )
        # attention_output = Multiply()([in_conv, attention])

        context_output_layer = create_context_module(
            in_conv,
            # attention_output,
            n_level_filters,
            dropout_rate=dropout_rate,  # in_cov -> attention
        )

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = []
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number], interpolation=interpolation
        )

        # Reg attention
        # attention = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(
        #     level_output_layers[level_number], up_sampling
        # )
        attention = Attention()([level_output_layers[level_number], up_sampling])
        attention = Multiply()([up_sampling, attention])
        # attention = LayerNormalization()(attention)

        concatenation_layer = Concatenate()(
            [
                level_output_layers[level_number],
                attention,
                # up_sampling,
            ]
        )
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number]
        )
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv2D(n_labels, (1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels - 1)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling2D(size=(2, 2), interpolation=interpolation)(
                output_layer
            )

    out_layer = Flatten()(output_layer)
    # out_layer = Dropout(dropout_rate)(out_layer)
    # out_layer = Dense(
    #     32,
    #     activation="relu",
    #     kernel_initializer="he_uniform",
    #     kernel_regularizer=kernel_regularizer,
    # )(out_layer)
    out_layer = Dense(1)(out_layer)

    model = Model(inputs=inputs, outputs=out_layer, name=name)

    # Allows for us to pass in a complete optimizer or incomplete with learning rate
    if callable(optimizer):
        optimizer = optimizer(learning_rate=initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model
