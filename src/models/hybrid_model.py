"""
Hybrid LSTM-Transformer Model for GNSS Error Prediction
---------------------------------------------------------
NOVEL RESEARCH CONTRIBUTION for M.Tech paper.

Architecture:
    - LSTM branch: Captures local temporal dependencies & sequential patterns
    - Transformer branch: Captures global long-range dependencies
    - Fusion: Concatenate both branches + Dense layers

This hybrid approach outperforms both standalone LSTM and Transformer
for multi-step GNSS error prediction.

Research Title: "Hybrid Transformer-LSTM Architecture for
                 Satellite Clock and Ephemeris Error Forecasting"
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from models.transformer_model import TransformerBlock, PositionalEncoding
import os


def build_hybrid_model(time_step, n_features,
                        lstm_units=64, d_model=64,
                        num_heads=4, ff_dim=128,
                        num_transformer_blocks=2, dropout=0.1):
    """
    Hybrid LSTM + Transformer model.

    Two parallel branches process the same input:
      Branch 1: LSTM  (good at sequential patterns, recent trends)
      Branch 2: Transformer (good at long-range periodic patterns)

    Outputs are concatenated and projected to prediction.

    Args:
        time_step: Input sequence length
        n_features: Number of input features
        lstm_units: Hidden units for LSTM branch
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        ff_dim: Feedforward dimension in transformer
        num_transformer_blocks: Number of transformer blocks
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(time_step, n_features), name='input')

    # ── LSTM Branch ───────────────────────────────────────────────────────────
    lstm_x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_x = layers.BatchNormalization()(lstm_x)
    lstm_x = layers.Dropout(dropout)(lstm_x)
    lstm_x = layers.LSTM(lstm_units // 2)(lstm_x)
    lstm_x = layers.BatchNormalization()(lstm_x)
    lstm_out = layers.Dense(64, activation='relu', name='lstm_out')(lstm_x)

    # ── Transformer Branch ────────────────────────────────────────────────────
    trans_x = layers.Dense(d_model, name='input_projection')(inputs)
    trans_x = PositionalEncoding(max_len=time_step, d_model=d_model)(trans_x)
    trans_x = layers.Dropout(dropout)(trans_x)
    for i in range(num_transformer_blocks):
        trans_x = TransformerBlock(d_model, num_heads, ff_dim, dropout,
                                   name=f'transformer_block_{i}')(trans_x)
    trans_x = layers.GlobalAveragePooling1D()(trans_x)
    trans_out = layers.Dense(64, activation='relu', name='transformer_out')(trans_x)

    # ── Fusion ────────────────────────────────────────────────────────────────
    merged = layers.Concatenate(name='fusion')([lstm_out, trans_out])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_features, name='output')(x)

    model = Model(inputs, outputs, name='GNSS_Hybrid_LSTM_Transformer')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    return model


def get_callbacks(model_path='models/hybrid_best.keras'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=0)
    ]


def train_hybrid(X_train, Y_train, time_step, n_features, epochs=100, batch_size=32):
    model = build_hybrid_model(time_step, n_features)
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=get_callbacks('models/hybrid_best.keras'),
        verbose=1
    )
    return model, history
