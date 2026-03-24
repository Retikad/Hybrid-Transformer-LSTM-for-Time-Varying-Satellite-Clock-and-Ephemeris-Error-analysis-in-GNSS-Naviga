"""
Transformer Model for GNSS Error Prediction
---------------------------------------------
Multi-head self-attention transformer encoder for time series.
Best model for M.Tech research paper - captures long-range dependencies.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
Applied to GNSS error forecasting.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, max_len=500, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.cast(tf.range(seq_len), dtype=tf.float32)
        dims = tf.cast(tf.range(self.d_model), dtype=tf.float32)

        angles = positions[:, tf.newaxis] / tf.pow(
            10000.0, (2 * (dims[tf.newaxis, :] // 2)) / tf.cast(self.d_model, tf.float32)
        )
        sin_enc = tf.math.sin(angles[:, 0::2])
        cos_enc = tf.math.cos(angles[:, 1::2])

        # Interleave sin and cos
        encoding = tf.reshape(
            tf.stack([sin_enc, cos_enc], axis=2),
            [seq_len, self.d_model]
        )
        return x + encoding[tf.newaxis, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


class TransformerBlock(layers.Layer):
    """Single transformer encoder block with multi-head attention + FFN."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head self attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        # Feed forward network
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)


def build_transformer_model(time_step, n_features,
                             d_model=64, num_heads=4,
                             ff_dim=128, num_blocks=3, dropout=0.1):
    """
    Build Transformer encoder model for GNSS error prediction.

    Args:
        time_step: Sequence length (lookback window)
        n_features: Input feature dimensions
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network hidden dim
        num_blocks: Number of transformer blocks
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(time_step, n_features))

    # Project input to d_model dimensions
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=time_step, d_model=d_model)(x)
    x = layers.Dropout(dropout)(x)

    # Stack transformer blocks
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

    # Global average pooling over sequence
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_features)(x)

    model = Model(inputs, outputs, name='GNSS_Transformer')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    return model


def get_callbacks(model_path='models/transformer_best.keras'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=0)
    ]


def train_transformer(X_train, Y_train, time_step, n_features, epochs=100, batch_size=32):
    model = build_transformer_model(time_step, n_features)
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=get_callbacks('models/transformer_best.keras'),
        verbose=1
    )
    return model, history
