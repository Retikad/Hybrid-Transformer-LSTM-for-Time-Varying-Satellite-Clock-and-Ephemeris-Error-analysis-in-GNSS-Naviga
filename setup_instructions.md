# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/Retikad/Hybrid-Transformer-LSTM-for-Time-Varying-Satellite-Clock-and-Ephemeris-Error-analysis-in-GNSS-Naviga.git
cd GNSS_Project
```

### 2. Create a Virtual Environment (Recommended)

#### Using venv:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Using conda:
```bash
conda create -n gnss-env python=3.8
conda activate gnss-env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

- **models/**: Contains model definitions and trained models
  - `lstm_model.py`: LSTM model implementation
  - `transformer_model.py`: Transformer model implementation
  - `hybrid_model.py`: Hybrid model combining LSTM and Transformer
  - `preprocessing.py`: Data preprocessing utilities
  - `evaluation.py`: Model evaluation metrics
  - `*_best.keras`: Trained model weights

- **data/**: Contains data generation and raw data files
  - `generate_data.py`: Script to generate synthetic GNSS error data
  - `gnss_errors.csv`: Dataset

- **results/**: Contains output results, metrics, and visualizations
  - `metrics_table.csv`: Model performance metrics
  - `validity_period_rmse.csv`: RMSE analysis by validity period

- **train.py**: Main training script
- **demo.py**: Demo/inference script
- **requirements.txt**: Project dependencies

## Usage

### Training Models
```bash
python train.py
```

### Running Demo/Inference
```bash
python demo.py
```

### Generating Data
```bash
python data/generate_data.py
```

## Contributing
Please fork the repository and submit pull requests for any improvements.

## License
See the repository for license information.
