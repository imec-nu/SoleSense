"""SoleFormer model architecture.

This module contains the TensorFlow/Keras implementation of SoleFormer used
for joint sound event detection (SED) and direction-of-arrival (DOA)
classification.

Expected input shape (example):
    (time_frames, frequency_bins, channels) = (61, 256, 6)

Outputs:
    full_model:
        - sed_output: binary sound-event probability
        - doa_output: 8-class direction probability
        - embedding_output: pooled latent representation

    train_model:
        - sed_output
        - doa_output
"""

from __future__ import annotations

from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling2D,
    Multiply,
    Reshape,
)
from tensorflow.keras.optimizers import Adam


DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (61, 256, 6)
DEFAULT_NUM_DOA_CLASSES = 8


def soleformer_block(
    x: tf.Tensor,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.1,
    kernel_size: int = 31,
    block_idx: int = 0,
) -> tf.Tensor:
    """Apply one SoleFormer block.
    """
    prefix = f"soleformer_{block_idx}"
    feature_dim = x.shape[-1]
    if feature_dim is None:
        raise ValueError("The input feature dimension must be statically known.")

    feature_dim = int(feature_dim)

    # First half-step feed-forward module.
    ff1 = Dense(
        ff_dim,
        activation="relu",
        name=f"{prefix}_ff1_dense1",
    )(x)
    ff1 = Dropout(dropout, name=f"{prefix}_ff1_dropout")(ff1)
    ff1 = Dense(feature_dim, name=f"{prefix}_ff1_dense2")(ff1)
    ff1 = Lambda(
        lambda tensor: 0.5 * tensor,
        name=f"{prefix}_ff1_half_step",
    )(ff1)
    x = Add(name=f"{prefix}_ff1_add")([x, ff1])

    # Global branch: multi-head self-attention.
    attention_input = LayerNormalization(
        epsilon=1e-6,
        name=f"{prefix}_attention_norm",
    )(x)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout,
        name=f"{prefix}_attention",
    )(attention_input, attention_input)
    attention_output = Dropout(
        dropout,
        name=f"{prefix}_attention_dropout",
    )(attention_output)

    # Local branch: pointwise convolution, GLU, depthwise convolution,
    # batch normalization, Swish activation, and output projection.
    conv_input = LayerNormalization(
        epsilon=1e-6,
        name=f"{prefix}_conv_norm",
    )(x)
    conv_projection = Conv1D(
        filters=2 * feature_dim,
        kernel_size=1,
        padding="same",
        name=f"{prefix}_conv_pointwise_in",
    )(conv_input)

    conv_u = Lambda(
        lambda tensor: tensor[:, :, :feature_dim],
        name=f"{prefix}_glu_u",
    )(conv_projection)
    conv_v = Lambda(
        lambda tensor: tensor[:, :, feature_dim:],
        name=f"{prefix}_glu_v",
    )(conv_projection)
    conv_v = Activation("sigmoid", name=f"{prefix}_glu_sigmoid")(conv_v)
    conv_glu = Multiply(name=f"{prefix}_glu")([conv_u, conv_v])

    conv_output = DepthwiseConv1D(
        kernel_size=kernel_size,
        padding="same",
        name=f"{prefix}_depthwise_conv",
    )(conv_glu)
    conv_output = BatchNormalization(
        name=f"{prefix}_depthwise_batch_norm",
    )(conv_output)
    conv_output = Activation("swish", name=f"{prefix}_swish")(conv_output)
    conv_output = Conv1D(
        filters=feature_dim,
        kernel_size=1,
        padding="same",
        name=f"{prefix}_conv_pointwise_out",
    )(conv_output)
    conv_output = Dropout(
        dropout,
        name=f"{prefix}_conv_dropout",
    )(conv_output)

    # Merge the global and local branches, then apply the residual connection.
    merged = Add(name=f"{prefix}_branch_merge")(
        [attention_output, conv_output]
    )
    x = Add(name=f"{prefix}_residual_merge")([x, merged])

    # Second half-step feed-forward module.
    ff2 = Dense(
        ff_dim,
        activation="relu",
        name=f"{prefix}_ff2_dense1",
    )(x)
    ff2 = Dropout(dropout, name=f"{prefix}_ff2_dropout")(ff2)
    ff2 = Dense(feature_dim, name=f"{prefix}_ff2_dense2")(ff2)
    ff2 = Lambda(
        lambda tensor: 0.5 * tensor,
        name=f"{prefix}_ff2_half_step",
    )(ff2)
    x = Add(name=f"{prefix}_ff2_add")([x, ff2])

    return LayerNormalization(
        epsilon=1e-6,
        name=f"{prefix}_output_norm",
    )(x)


def masked_categorical_crossentropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Categorical cross-entropy that ignores unlabeled DOA samples.
    """
    per_sample_loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(
        tf.reduce_any(y_true > 0, axis=-1),
        dtype=per_sample_loss.dtype,
    )
    valid_sample_count = tf.maximum(
        tf.reduce_sum(mask),
        tf.cast(1.0, mask.dtype),
    )
    return tf.reduce_sum(per_sample_loss * mask) / valid_sample_count


def build_soleformer_seld(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_layers: int = 1,
    head_size: int = 32,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout_rate: float = 0.15,
    fnn_units: Sequence[int] = (128,),
    num_doa_classes: int = DEFAULT_NUM_DOA_CLASSES,
) -> Tuple[Model, Model]:
    """Build the complete SoleFormer SELD network.
    """
    if len(fnn_units) == 0:
        raise ValueError("fnn_units must contain at least one hidden dimension.")
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    spectrogram_input = Input(
        shape=input_shape,
        name="spectrogram_input",
    )

    x = spectrogram_input

    # Convolutional front end.
    for block_index, pool_size in enumerate(((1, 8), (1, 8), (1, 2)), start=1):
        x = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            name=f"frontend_conv_{block_index}",
        )(x)
        x = BatchNormalization(
            name=f"frontend_batch_norm_{block_index}",
        )(x)
        x = Activation(
            "relu",
            name=f"frontend_relu_{block_index}",
        )(x)
        x = MaxPooling2D(
            pool_size=pool_size,
            name=f"frontend_pool_{block_index}",
        )(x)
        x = Dropout(
            dropout_rate,
            name=f"frontend_dropout_{block_index}",
        )(x)

    # Preserve the time dimension and flatten frequency/channel features.
    x = Reshape(
        (input_shape[0], -1),
        name="frontend_sequence_reshape",
    )(x)

    for layer_index in range(num_layers):
        x = soleformer_block(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
            kernel_size=31,
            block_idx=layer_index,
        )

    embedding_output = GlobalAveragePooling1D(
        name="embedding_output",
    )(x)

    # Sound event detection head.
    sed_output = Dense(
        fnn_units[0],
        activation="relu",
        name="sed_dense",
    )(embedding_output)
    sed_output = Dropout(
        dropout_rate,
        name="sed_dropout",
    )(sed_output)
    sed_output = Dense(
        1,
        activation="sigmoid",
        name="sed_output",
    )(sed_output)

    # Direction-of-arrival classification head.
    doa_output = Dense(
        fnn_units[0],
        activation="relu",
        name="doa_dense",
    )(embedding_output)
    doa_output = Dropout(
        dropout_rate,
        name="doa_dropout",
    )(doa_output)
    doa_output = Dense(
        num_doa_classes,
        activation="softmax",
        name="doa_output",
    )(doa_output)

    full_model = Model(
        inputs=spectrogram_input,
        outputs=[sed_output, doa_output, embedding_output],
        name="soleformer_seld_full",
    )
    train_model = Model(
        inputs=spectrogram_input,
        outputs=[sed_output, doa_output],
        name="soleformer_seld",
    )

    return full_model, train_model


def compile_soleformer(
    model: Model,
    learning_rate: float = 8e-5,
    sed_loss_weight: float = 10.0,
    doa_loss_weight: float = 10.0,
) -> Model:
    """Compile a SoleFormer training model."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[
            tf.keras.losses.BinaryCrossentropy(),
            masked_categorical_crossentropy,
        ],
        loss_weights=[sed_loss_weight, doa_loss_weight],
    )
    return model



if __name__ == "__main__":
    full_model, train_model = build_soleformer_seld()
    compile_soleformer(train_model)
    train_model.summary()