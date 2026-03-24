"""
Evaluation Metrics for GNSS Error Prediction
----------------------------------------------
Computes RMSE, MAE, MAPE, error distribution analysis,
and ISRO-specific validity period evaluation.

ISRO Requirement:
    Error distribution should be close to normal distribution.
    Validity periods: 15min, 30min, 1hr, 2hr, 24hr
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def compute_metrics(y_true, y_pred, feature_names=None):
    """
    Compute comprehensive prediction metrics.

    Returns:
        dict with RMSE, MAE, MAPE, R2 per feature
    """
    if y_pred.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_features = y_true.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    results = {}
    for i, name in enumerate(feature_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        errors = yt - yp

        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)

        # MAPE (avoid division by zero)
        mask = np.abs(yt) > 1e-6
        mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100 if mask.sum() > 0 else np.nan

        # R² score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Normality test (Shapiro-Wilk on residuals)
        sw_stat, sw_p = stats.shapiro(errors[:min(5000, len(errors))])

        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Error_Mean': np.mean(errors),
            'Error_Std': np.std(errors),
            'Shapiro_W': sw_stat,
            'Shapiro_p': sw_p,
            'Normal_dist': 'YES' if sw_p > 0.05 else 'NO'
        }

    return results


def evaluate_validity_periods(model, X_test, Y_test_scaled, scalers,
                               feature_cols, interval_minutes=15):
    """
    ISRO Requirement: Evaluate prediction accuracy at different validity periods.
    Periods: 15min, 30min, 1hr, 2hr, 24hr

    Returns:
        DataFrame with RMSE per validity period per feature
    """
    validity_periods = {
        '15min': 1,
        '30min': 2,
        '1hr': 4,
        '2hr': 8,
        '24hr': 96
    }

    # Predict all test points
    y_pred_scaled = model.predict(X_test, verbose=0)

    results = {}
    for period_name, steps in validity_periods.items():
        if steps > len(Y_test_scaled):
            continue
        yt = Y_test_scaled[:steps]
        yp = y_pred_scaled[:steps]
        rmse_vals = {}
        for i, col in enumerate(feature_cols):
            rmse_vals[col] = np.sqrt(mean_squared_error(yt[:, i], yp[:, i]))
        results[period_name] = rmse_vals

    df_results = pd.DataFrame(results).T
    return df_results


def plot_predictions(y_true, y_pred, feature_names, model_name, save_dir='results'):
    """Plot actual vs predicted for all features."""
    os.makedirs(save_dir, exist_ok=True)
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(y_true[:, i], label='Actual', color='steelblue', alpha=0.8, linewidth=1)
        ax.plot(y_pred[:, i], label='Predicted', color='tomato', alpha=0.8, linewidth=1, linestyle='--')
        ax.set_title(f'{model_name} — {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (15-min intervals)')
        ax.set_ylabel('Scaled Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_error_distribution(y_true, y_pred, feature_names, model_name, save_dir='results'):
    """
    ISRO Requirement: Plot error distribution vs Normal distribution.
    Closer to Normal = better model.
    """
    os.makedirs(save_dir, exist_ok=True)
    errors = y_true - y_pred
    n_features = len(feature_names)

    fig, axes = plt.subplots(2, n_features, figsize=(6 * n_features, 10))
    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(feature_names):
        err = errors[:, i]

        # Histogram + Normal fit
        ax = axes[0, i]
        mu, sigma = np.mean(err), np.std(err)
        ax.hist(err, bins=50, density=True, color='steelblue', alpha=0.7, label='Error Dist.')
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        sw_stat, sw_p = stats.shapiro(err[:min(5000, len(err))])
        ax.set_title(f'{name}\nShapiro-Wilk p={sw_p:.4f} ({" ✓ Normal" if sw_p > 0.05 else "✗ Non-normal"})',
                     fontsize=11)
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Q-Q Plot
        ax2 = axes[1, i]
        stats.probplot(err, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot — {name}')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} — Error Distribution Analysis (ISRO Requirement)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_training_history(histories, model_names, save_dir='results'):
    """Plot training loss curves for all models."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue', 'tomato', 'seagreen']

    for i, (history, name) in enumerate(zip(histories, model_names)):
        c = colors[i % len(colors)]
        axes[0].plot(history.history['loss'], label=f'{name} Train', color=c)
        axes[0].plot(history.history['val_loss'], label=f'{name} Val', color=c, linestyle='--')
        axes[1].plot(history.history['mae'], label=f'{name} Train', color=c)
        axes[1].plot(history.history['val_mae'], label=f'{name} Val', color=c, linestyle='--')

    for ax, title in zip(axes, ['Loss (MSE)', 'MAE']):
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training History — All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(comparison_df, save_dir='results'):
    """Bar chart comparing RMSE across all models."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', linewidth=0.5)
    ax.set_title('Model Comparison — RMSE by Feature', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.legend(title='Feature', bbox_to_anchor=(1.05, 1))
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_metrics_table(all_metrics, save_dir='results'):
    """Save metrics as CSV for paper tables."""
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for model_name, feat_metrics in all_metrics.items():
        for feat, m in feat_metrics.items():
            rows.append({'Model': model_name, 'Feature': feat, **m})
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, 'metrics_table.csv')
    df.to_csv(path, index=False)
    print(f"\nMetrics saved to {path}")
    print(df[['Model', 'Feature', 'RMSE', 'MAE', 'R2', 'Normal_dist']].to_string(index=False))
    return df
