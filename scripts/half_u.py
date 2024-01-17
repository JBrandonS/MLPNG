from keras.layers import (
    Add,
    Dense,
    Dropout,
    Flatten,
    MultiHeadAttention,
    Multiply,
    RandomFlip,
)
from keras.models import Model
from keras.optimizers.legacy import Adam

from tf_utils import (
    dice_coefficient_loss,
    ReflectionPadding2D,
    create_context_module,
    create_convolution_block,
)


def half_u(
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
):
    current_layer = inputs

    if flip:
        current_layer = RandomFlip()(current_layer)

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
        attention = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim)(
            in_conv, in_conv
        )
        attention_output = Multiply()([in_conv, attention])

        context_output_layer = create_context_module(
            # in_conv,
            attention_output,
            n_level_filters,
            dropout_rate=dropout_rate,  # in_cov -> attention
        )

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # FF network
    out_layer = Flatten()(current_layer)
    out_layer = Dropout(dropout_rate)(out_layer)
    out_layer = Dense(512, activation="sigmoid")(out_layer)
    out_layer = Dropout(dropout_rate)(out_layer)
    out_layer = Dense(
        128,
        activation="relu",
        kernel_initializer="he_uniform",
        kernel_regularizer=kernel_regularizer,
    )(out_layer)
    out_layer = Dense(
        32,
        activation="relu",
        kernel_initializer="he_uniform",
        kernel_regularizer=kernel_regularizer,
    )(out_layer)
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
