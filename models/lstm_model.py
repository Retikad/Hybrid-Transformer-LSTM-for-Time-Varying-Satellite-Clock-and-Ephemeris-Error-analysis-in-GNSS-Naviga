"""
LSTM Model for GNSS Error Prediction
--------------------------------------
Stacked LSTM with dropout for time series forecasting.
Baseline model for M.Tech research comparison.
"""

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os


def build_lstm_model(time_step, n_features, units=64):
    """
    Stacked LSTM model.

    Args:
        time_step: Input sequence length
        n_features: Number of input features
        units: LSTM hidden units

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(time_step, n_features)),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(units // 2, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(units // 4),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(n_features)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model


def get_callbacks(model_path='models/lstm_best.keras'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=0)
    ]


def train_lstm(X_train, Y_train, time_step, n_features, epochs=100, batch_size=32):
    model = build_lstm_model(time_step, n_features)
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=get_callbacks('models/lstm_best.keras'),
        verbose=1
    )
    return model, history
