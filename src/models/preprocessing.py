"""
GNSS Data Preprocessing Utilities
----------------------------------
Handles data loading, cleaning, feature engineering,
and sequence creation for deep learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_and_preprocess(csv_path, satellite_id=None, feature_cols=None):
    """
    Load GNSS error dataset and preprocess it.

    Args:
        csv_path: Path to CSV file
        satellite_id: Filter for specific satellite (None = all combined)
        feature_cols: List of columns to use as features

    Returns:
        df: Processed dataframe
        scalers: Dict of fitted scalers for inverse transform
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    if satellite_id:
        df = df[df['satellite_id'] == satellite_id].reset_index(drop=True)
        print(f"Filtered for satellite: {satellite_id}, rows: {len(df)}")

    if feature_cols is None:
        feature_cols = ['clock_error_ns', 'total_eph_error_m',
                        'ephemeris_error_x_m', 'ephemeris_error_y_m', 'ephemeris_error_z_m']

    # Keep only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Add time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    print(f"Features used: {feature_cols}")
    print(f"Data shape: {df[feature_cols].shape}")
    print(f"Missing values: {df[feature_cols].isnull().sum().sum()}")

    # Scale features
    scalers = {}
    df_scaled = df.copy()
    for col in feature_cols:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_scaled[col + '_scaled'] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df_scaled, scalers, feature_cols


def create_sequences(data, time_step=48, forecast_step=1):
    """
    Create sliding window sequences for time series prediction.

    Args:
        data: numpy array (n_samples, n_features)
        time_step: lookback window (48 = 12 hours at 15-min intervals)
        forecast_step: how many steps ahead to predict

    Returns:
        X: (n_sequences, time_step, n_features)
        Y: (n_sequences, n_features)
    """
    X, Y = [], []
    for i in range(len(data) - time_step - forecast_step + 1):
        X.append(data[i:i + time_step])
        Y.append(data[i + time_step + forecast_step - 1])
    return np.array(X), np.array(Y)


def train_test_split_temporal(X, Y, test_ratio=0.125):
    """
    Split sequences temporally (last N% for testing = Day 8 simulation).
    test_ratio=0.125 → last 1/8 of data = simulated 8th day
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], Y[:split], Y[split:]


def save_scalers(scalers, path='models/scalers.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scalers, path)
    print(f"Scalers saved to {path}")


def load_scalers(path='models/scalers.pkl'):
    return joblib.load(path)


def inverse_transform_predictions(predictions, scaler, feature_idx=0):
    """Inverse transform scaled predictions back to original units."""
    # Create dummy array for inverse transform
    dummy = np.zeros((len(predictions), 1))
    dummy[:, 0] = predictions[:, feature_idx] if predictions.ndim > 1 else predictions
    return scaler.inverse_transform(dummy)[:, 0]
