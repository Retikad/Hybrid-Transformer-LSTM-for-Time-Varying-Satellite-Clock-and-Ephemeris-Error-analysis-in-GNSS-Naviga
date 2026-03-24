"""
GNSS Synthetic Data Generator
------------------------------
Generates 7-day synthetic GNSS satellite clock and ephemeris error data
at 15-minute intervals for training and testing models.

Replace this with real ISRO dataset when available.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 7 days * 24 hours * 4 samples/hour = 672 samples
n_samples = 7 * 24 * 4
time_index = pd.date_range(start='2024-01-01 00:00:00', periods=n_samples, freq='15min')

def generate_gnss_errors(n, noise_std=0.3):
    """Simulate realistic GNSS errors using sinusoidal + trend + noise."""
    t = np.arange(n)
    # Orbital period (~12h for MEO) -> 48 samples per period
    orbit_period = 48
    # Daily drift
    daily_drift = 0.05 * np.sin(2 * np.pi * t / (4 * 24))
    # Sub-orbital variation
    orbital = 0.8 * np.sin(2 * np.pi * t / orbit_period)
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 0.01, n))
    random_walk = random_walk / random_walk.std() * 0.2
    # White noise
    noise = np.random.normal(0, noise_std, n)
    return daily_drift + orbital + random_walk + noise

# Satellite IDs (mix of GEO and MEO)
satellites = {
    'GEO_01': {'clock_std': 0.2, 'eph_std': 0.15},
    'GEO_02': {'clock_std': 0.25, 'eph_std': 0.18},
    'MEO_01': {'clock_std': 0.3, 'eph_std': 0.22},
    'MEO_02': {'clock_std': 0.28, 'eph_std': 0.20},
    'MEO_03': {'clock_std': 0.35, 'eph_std': 0.25},
}

rows = []
for sat_id, params in satellites.items():
    clock_errors = generate_gnss_errors(n_samples, noise_std=params['clock_std'])
    eph_x_errors = generate_gnss_errors(n_samples, noise_std=params['eph_std'])
    eph_y_errors = generate_gnss_errors(n_samples, noise_std=params['eph_std'] * 0.8)
    eph_z_errors = generate_gnss_errors(n_samples, noise_std=params['eph_std'] * 0.9)

    for i in range(n_samples):
        rows.append({
            'timestamp': time_index[i],
            'satellite_id': sat_id,
            'satellite_type': 'GEO' if 'GEO' in sat_id else 'MEO',
            'clock_error_ns': clock_errors[i],          # nanoseconds
            'ephemeris_error_x_m': eph_x_errors[i],    # meters
            'ephemeris_error_y_m': eph_y_errors[i],
            'ephemeris_error_z_m': eph_z_errors[i],
            'total_eph_error_m': np.sqrt(eph_x_errors[i]**2 + eph_y_errors[i]**2 + eph_z_errors[i]**2)
        })

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
df.to_csv('data/gnss_errors.csv', index=False)
print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head(10))
print("\nSatellites:", df['satellite_id'].unique())
print("Date range:", df['timestamp'].min(), "to", df['timestamp'].max())
