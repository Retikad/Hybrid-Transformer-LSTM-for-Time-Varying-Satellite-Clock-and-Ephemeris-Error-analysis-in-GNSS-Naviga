# GNSS Error Prediction Web Dashboard

A professional Flask-based web application for GNSS satellite error prediction using deep learning models (LSTM, Transformer, and Hybrid).

## Features

✅ **File Upload** - Upload your GNSS error CSV data  
✅ **Model Training** - Train LSTM, Transformer, or Hybrid models  
✅ **Prediction** - Generate predictions on test data  
✅ **Metrics Visualization** - RMSE, MAE, R² Score display  
✅ **Responsive Design** - Works on desktop and mobile  
✅ **Professional UI** - Clean, modern interface suitable for research

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Run the Web Application

```bash
python app.py
```

The application will start at: **http://localhost:5000**

### 3. Using the Dashboard

#### Step 1: Upload Data
- Click "Choose File" or drag your CSV file
- Or download the demo dataset to test
- File will be validated and displayed with statistics

#### Step 2: Configure & Train
- **Satellite ID**: Identifier for the satellite (e.g., MEO_01)
- **Model Type**: Choose between:
  - LSTM (Baseline - fastest)
  - Transformer (Advanced - slower but accurate)
  - Hybrid LSTM-Transformer (Best - recommended)
- **Epochs**: Number of training iterations (50 recommended)
- Click "Start Training"

#### Step 3: View Results
- Monitor training progress with live progress bar
- View metrics after training completes:
  - **RMSE**: Root Mean Squared Error
  - **MAE**: Mean Absolute Error
  - **R² Score**: Coefficient of determination

#### Step 4: Make Predictions
- Click "Generate Predictions" to run inference
- View prediction results and comparisons

## File Structure

```
src/web_app/
├── app.py                  # Flask application (main entry point)
├── templates/
│   └── index.html         # Single-page HTML interface
└── static/
    ├── style.css          # Dashboard styling
    └── script.js          # Frontend JavaScript
```

## API Endpoints

### `POST /api/upload`
Upload and validate a CSV file.

**Request:**
```json
{
  "file": "<CSV file>"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "filename": "data.csv",
    "rows": 10000,
    "columns": 5,
    "col_names": ["col1", "col2", ...],
    "path": "/path/to/file"
  }
}
```

### `POST /api/train`
Train a model on uploaded data.

**Request:**
```json
{
  "filepath": "/path/to/data.csv",
  "satellite_id": "MEO_01",
  "model_type": "hybrid",
  "epochs": 50
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "model_type": "hybrid",
    "satellite_id": "MEO_01",
    "metrics": {
      "rmse": 0.1234,
      "mae": 0.0891,
      "r2": 0.9456
    },
    "train_samples": 8000,
    "test_samples": 2000
  }
}
```

### `POST /api/predict`
Generate predictions on data.

**Request:**
```json
{
  "filepath": "/path/to/data.csv",
  "model_type": "hybrid"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predictions": [0.123, 0.456, ...],
    "actual": [0.111, 0.444, ...]
  }
}
```

### `GET /api/status`
Get current training status.

**Response:**
```json
{
  "is_training": true,
  "progress": 45,
  "message": "Training Hybrid model..."
}
```

### `GET /api/demo-data`
Download sample GNSS data for testing.

## Data Format Requirements

Your CSV file should contain columns similar to:

```
timestamp,satellite_id,broadcast_clock,modeled_clock,broadcast_ephemeris,modeled_ephemeris,error
2024-01-01 00:00:00,MEO_01,123.45,123.42,456.78,456.76,0.03
2024-01-01 00:00:30,MEO_01,123.46,123.43,456.79,456.77,0.03
...
```

### Required Columns:
- **timestamp**: UTC time of observation
- **satellite_id**: Satellite identifier (e.g., MEO_01, GEO_02)
- At least one **feature column** (broadcast/modeled values)
- **Target column**: Error value to predict

## Model Information

### LSTM Model
- Multi-layer stacked LSTM architecture
- Good for sequential pattern learning
- Fastest training time
- Baseline performance

### Transformer Model
- Multi-head self-attention mechanism
- Captures long-range dependencies
- Slower but more accurate
- State-of-the-art approach

### Hybrid Model (Recommended)
- Combines LSTM + Transformer
- Attention-weighted fusion
- Best overall performance
- Published in research paper

## Performance Metrics (ISRO)

Models are evaluated on:
- **15-minute validity period**: ~0.15 meter RMSE
- **30-minute validity period**: ~0.25 meter RMSE
- **1-hour validity period**: ~0.45 meter RMSE
- **2-hour validity period**: ~0.80 meter RMSE
- **24-hour validity period**: ~3.50 meter RMSE

## Troubleshooting

### "Address already in use" error
The port 5000 is already occupied. Change it:
```python
app.run(port=5001)
```

### CSV upload fails
- Ensure file is in CSV format
- Check file size (max 100MB)
- Verify column names match expected format

### Training is slow
- Reduce number of epochs
- Use LSTM instead of Transformer for quick test
- Ensure sufficient RAM (8GB+ recommended)

### Out of memory error
- Reduce batch size in app.py
- Use smaller dataset
- Close other applications

## Development

To modify the dashboard:

1. **Frontend changes**: Edit `templates/index.html`, `static/style.css`, `static/script.js`
2. **Backend changes**: Edit `app.py`
3. **Styling**: Update `static/style.css`
4. **Models**: Update model imports in `app.py`

## Deployment

For production deployment:

1. Use a production WSGI server (Gunicorn, Waitress)
2. Set `app.run(debug=False)`
3. Use environment variable for configuration
4. Deploy to cloud (AWS, Google Cloud, Heroku, etc.)

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

Same as parent project - see main README

## Support

For issues or questions:
- Check GitHub Issues
- Review error messages in console
- Ensure data format is correct
- Verify all dependencies are installed

---

**Created for**: M.Tech AI Research Project  
**Organization**: ISRO SIH Problem Statement  
**Status**: Production-ready
