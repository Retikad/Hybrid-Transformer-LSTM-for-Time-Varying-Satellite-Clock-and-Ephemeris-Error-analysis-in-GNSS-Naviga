"""
GNSS Error Prediction — Quick Demo
=====================================
Runs a fast demo version with fewer epochs to verify everything works.
Use train.py for full training (more epochs = better accuracy).

Usage:
    python demo.py
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

# Generate data first
print("Generating synthetic GNSS data...")
exec(open('data/generate_data.py').read())

from models.preprocessing import (
    load_and_preprocess, create_sequences, train_test_split_temporal, save_scalers
)
from models.lstm_model import train_lstm
from models.hybrid_model import train_hybrid
from models.evaluation import (
    compute_metrics, plot_predictions, plot_error_distribution,
    plot_training_history, save_metrics_table
)

# Fast config for demo
DATA_PATH = 'data/gnss_errors.csv'
SATELLITE_ID = 'MEO_01'
TIME_STEP = 32
EPOCHS = 15      # Fast demo - use 100 in train.py for real results
BATCH_SIZE = 64

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("\nLoading data...")
df_scaled, scalers, feature_cols = load_and_preprocess(
    DATA_PATH, satellite_id=SATELLITE_ID
)
save_scalers(scalers, 'models/scalers.pkl')

scaled_cols = [f + '_scaled' for f in feature_cols]
data_matrix = df_scaled[scaled_cols].values

print(f"\nCreating sequences...")
X, Y = create_sequences(data_matrix, time_step=TIME_STEP)
X_train, X_test, Y_train, Y_test = train_test_split_temporal(X, Y)
n_features = X.shape[2]

print(f"\nTraining LSTM (demo: {EPOCHS} epochs)...")
lstm_model, lstm_history = train_lstm(X_train, Y_train, TIME_STEP, n_features, EPOCHS, BATCH_SIZE)

print(f"\nTraining Hybrid LSTM-Transformer (demo: {EPOCHS} epochs)...")
hybrid_model, hybrid_history = train_hybrid(X_train, Y_train, TIME_STEP, n_features, EPOCHS, BATCH_SIZE)

print("\nEvaluating models...")
y_pred_lstm = lstm_model.predict(X_test, verbose=0)
y_pred_hybrid = hybrid_model.predict(X_test, verbose=0)

metrics_lstm = compute_metrics(Y_test, y_pred_lstm, feature_cols)
metrics_hybrid = compute_metrics(Y_test, y_pred_hybrid, feature_cols)

all_metrics = {
    'LSTM': metrics_lstm,
    'Hybrid_LSTM_Transformer': metrics_hybrid
}

print("\nGenerating plots...")
plot_predictions(Y_test, y_pred_lstm, feature_cols, 'LSTM', 'results')
plot_predictions(Y_test, y_pred_hybrid, feature_cols, 'Hybrid_LSTM_Transformer', 'results')
plot_error_distribution(Y_test, y_pred_hybrid, feature_cols, 'Hybrid_LSTM_Transformer', 'results')
plot_training_history([lstm_history, hybrid_history],
                      ['LSTM', 'Hybrid'], 'results')
save_metrics_table(all_metrics, 'results')

print("\n" + "=" * 50)
print("  DEMO COMPLETE!")
print("  Results saved in: results/")
print("  Run train.py for full training with all 3 models.")
print("=" * 50)
