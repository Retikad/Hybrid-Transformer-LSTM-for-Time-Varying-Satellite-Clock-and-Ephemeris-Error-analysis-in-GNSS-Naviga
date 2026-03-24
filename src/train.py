"""
GNSS Satellite Error Prediction — Main Training Script
=========================================================
M.Tech Research Project | ISRO SIH Problem 25176

Models:
    1. LSTM (Baseline)
    2. Transformer (State-of-art)
    3. Hybrid LSTM-Transformer (Novel Contribution)

Usage:
    python train.py

Output:
    - Trained models in models/
    - Plots in results/
    - Metrics CSV in results/metrics_table.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.preprocessing import (
    load_and_preprocess, create_sequences,
    train_test_split_temporal, save_scalers
)
from models.lstm_model import train_lstm
from models.transformer_model import train_transformer
from models.hybrid_model import train_hybrid
from models.evaluation import (
    compute_metrics, plot_predictions, plot_error_distribution,
    plot_training_history, plot_model_comparison,
    save_metrics_table, evaluate_validity_periods
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = 'data/gnss_errors.csv'
SATELLITE_ID = 'MEO_01'           # Change to None for all satellites
TIME_STEP = 48                     # Lookback: 48 × 15min = 12 hours
EPOCHS = 60                        # Increase for better results (100+ recommended)
BATCH_SIZE = 32
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

FEATURE_COLS = [
    'clock_error_ns',
    'total_eph_error_m',
    'ephemeris_error_x_m',
    'ephemeris_error_y_m',
    'ephemeris_error_z_m'
]
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  GNSS Error Prediction — M.Tech Research")
    print("  ISRO SIH Problem 25176")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load & Preprocess Data ─────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    df_scaled, scalers, feature_cols = load_and_preprocess(
        DATA_PATH,
        satellite_id=SATELLITE_ID,
        feature_cols=FEATURE_COLS
    )
    save_scalers(scalers, os.path.join(MODELS_DIR, 'scalers.pkl'))

    # Build scaled feature matrix
    scaled_cols = [f + '_scaled' for f in feature_cols]
    data_matrix = df_scaled[scaled_cols].values
    print(f"Data matrix shape: {data_matrix.shape}")

    # ── 2. Create Sequences ───────────────────────────────────────────────────
    print(f"\n[2/5] Creating sequences (time_step={TIME_STEP})...")
    X, Y = create_sequences(data_matrix, time_step=TIME_STEP)
    X_train, X_test, Y_train, Y_test = train_test_split_temporal(X, Y, test_ratio=0.125)
    n_features = X.shape[2]

    print(f"Training: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Testing:  X={X_test.shape}, Y={Y_test.shape}")

    # ── 3. Train Models ───────────────────────────────────────────────────────
    print("\n[3/5] Training models...")

    print("\n--- Model 1: LSTM (Baseline) ---")
    lstm_model, lstm_history = train_lstm(
        X_train, Y_train, TIME_STEP, n_features, EPOCHS, BATCH_SIZE
    )

    print("\n--- Model 2: Transformer ---")
    transformer_model, trans_history = train_transformer(
        X_train, Y_train, TIME_STEP, n_features, EPOCHS, BATCH_SIZE
    )

    print("\n--- Model 3: Hybrid LSTM-Transformer (Novel) ---")
    hybrid_model, hybrid_history = train_hybrid(
        X_train, Y_train, TIME_STEP, n_features, EPOCHS, BATCH_SIZE
    )

    # ── 4. Evaluate Models ────────────────────────────────────────────────────
    print("\n[4/5] Evaluating models...")

    models = {
        'LSTM': lstm_model,
        'Transformer': transformer_model,
        'Hybrid_LSTM_Transformer': hybrid_model
    }
    histories = [lstm_history, trans_history, hybrid_history]
    model_names = list(models.keys())

    all_metrics = {}
    comparison_rmse = {}

    for model_name, model in models.items():
        print(f"\n>>> Evaluating {model_name}...")
        y_pred = model.predict(X_test, verbose=0)

        metrics = compute_metrics(Y_test, y_pred, feature_cols)
        all_metrics[model_name] = metrics

        # Print metrics
        print(f"{'Feature':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Normal?':>10}")
        print("-" * 65)
        rmse_vals = {}
        for feat, m in metrics.items():
            print(f"{feat:<30} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R2']:>8.4f} {m['Normal_dist']:>10}")
            rmse_vals[feat] = m['RMSE']
        comparison_rmse[model_name] = rmse_vals

        # Generate plots
        plot_predictions(Y_test, y_pred, feature_cols, model_name, RESULTS_DIR)
        plot_error_distribution(Y_test, y_pred, feature_cols, model_name, RESULTS_DIR)

    # ── 5. Summary Plots & Reports ────────────────────────────────────────────
    print("\n[5/5] Generating summary plots...")

    plot_training_history(histories, model_names, RESULTS_DIR)

    comparison_df = pd.DataFrame(comparison_rmse).T
    plot_model_comparison(comparison_df, RESULTS_DIR)

    metrics_df = save_metrics_table(all_metrics, RESULTS_DIR)

    # Validity period evaluation (ISRO requirement)
    print("\n--- ISRO Validity Period Evaluation (Hybrid Model) ---")
    validity_df = evaluate_validity_periods(
        hybrid_model, X_test, Y_test, scalers, feature_cols
    )
    print(validity_df)
    validity_df.to_csv(os.path.join(RESULTS_DIR, 'validity_period_rmse.csv'))

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("  Files generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    - {f}")
    print("=" * 60)

    return models, all_metrics


if __name__ == '__main__':
    main()
