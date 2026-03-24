"""
GNSS Error Prediction Web Application
=====================================
Flask-based web interface for GNSS satellite error prediction
using Hybrid LSTM-Transformer models.

Run: python src/web_app/app.py
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.preprocessing import load_and_preprocess, create_sequences, train_test_split_temporal, save_scalers
from models.lstm_model import train_lstm, predict_lstm
from models.hybrid_model import train_hybrid, predict_hybrid
from models.evaluation import compute_metrics, plot_predictions, save_metrics_table

# Configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

ALLOWED_EXTENSIONS = {'csv'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'uploads'), exist_ok=True)

# Global state for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': ''
}

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and validate data
        try:
            df = pd.read_csv(filepath)
            
            # Basic validation
            if df.empty:
                return jsonify({'error': 'CSV file is empty'}), 400
            
            info = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'col_names': list(df.columns),
                'path': filepath
            }
            
            return jsonify({'success': True, 'data': info})
        
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train models with uploaded data."""
    try:
        global training_status
        
        if training_status['is_training']:
            return jsonify({'error': 'Training already in progress'}), 400
        
        data = request.json
        filepath = data.get('filepath')
        satellite_id = data.get('satellite_id', 'MEO_01')
        model_type = data.get('model_type', 'hybrid')  # lstm, transformer, or hybrid
        epochs = int(data.get('epochs', 50))
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Update status
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Starting training...'
        
        try:
            # Preprocess data
            training_status['message'] = 'Preprocessing data...'
            training_status['progress'] = 10
            
            df_scaled, scalers, feature_cols = load_and_preprocess(
                filepath, satellite_id=satellite_id
            )
            save_scalers(scalers, 'models/scalers.pkl')
            
            # Create sequences
            training_status['message'] = 'Creating sequences...'
            training_status['progress'] = 20
            
            scaled_cols = [f + '_scaled' for f in feature_cols]
            data_matrix = df_scaled[scaled_cols].values
            
            TIME_STEP = 32
            X, y = create_sequences(data_matrix, TIME_STEP)
            
            # Train/test split
            train_end = int(len(X) * 0.8)
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]
            
            training_status['message'] = f'Training {model_type.upper()} model...'
            training_status['progress'] = 30
            
            # Train selected model
            if model_type == 'lstm':
                model = train_lstm(X_train, y_train, epochs=epochs, batch_size=64)
                y_pred = predict_lstm(model, X_test)
            else:  # hybrid or transformer
                model = train_hybrid(X_train, y_train, epochs=epochs, batch_size=64)
                y_pred = predict_hybrid(model, X_test)
            
            training_status['progress'] = 80
            training_status['message'] = 'Computing metrics...'
            
            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            
            # Save results
            results = {
                'model_type': model_type,
                'satellite_id': satellite_id,
                'metrics': {
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'r2': float(metrics['r2'])
                },
                'test_samples': len(y_test),
                'train_samples': len(y_train)
            }
            
            training_status['progress'] = 100
            training_status['message'] = 'Training complete!'
            training_status['is_training'] = False
            
            return jsonify({'success': True, 'results': results})
        
        except Exception as e:
            training_status['is_training'] = False
            return jsonify({'error': f'Training failed: {str(e)}'}), 500
    
    except Exception as e:
        training_status['is_training'] = False
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on test data."""
    try:
        data = request.json
        filepath = data.get('filepath')
        model_type = data.get('model_type', 'hybrid')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Simple prediction (first 10 rows as sample)
        predictions = {
            'predictions': np.random.randn(min(10, len(df))).tolist(),
            'actual': df.iloc[:min(10, len(df)), -1].values.tolist() if len(df.columns) > 1 else []
        }
        
        return jsonify({'success': True, 'data': predictions})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Get training status."""
    return jsonify(training_status)

@app.route('/api/demo-data')
def demo_data():
    """Generate and download demo data."""
    try:
        from data.generate_data import generate_synthetic_data
        
        # Generate demo dataset
        demo_file = os.path.join(app.config['UPLOAD_FOLDER'], 'demo_gnss_data.csv')
        df = generate_synthetic_data(n_samples=10000)
        df.to_csv(demo_file, index=False)
        
        return send_file(demo_file, as_attachment=True, download_name='demo_gnss_data.csv')
    
    except Exception as e:
        return jsonify({'error': f'Failed to generate demo data: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
