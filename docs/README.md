# GNSS Satellite Error Prediction
## M.Tech Research Project | ISRO SIH Problem 25176

---

## Project Overview

Predicts time-varying **satellite clock and ephemeris errors** between uploaded (broadcast) and modeled GNSS values using three deep learning models:

| Model | Type | Purpose |
|-------|------|---------|
| LSTM | Baseline | Sequential pattern learning |
| Transformer | State-of-art | Long-range dependency capture |
| **Hybrid LSTM-Transformer** | **Novel Contribution** | **Best accuracy (Research Paper)** |

---

## Folder Structure

```
GNSS_Project/
│
├── data/
│   ├── generate_data.py       # Generates synthetic dataset
│   └── gnss_errors.csv        # Dataset (auto-generated or replace with real data)
│
├── models/
│   ├── preprocessing.py       # Data loading, scaling, sequence creation
│   ├── lstm_model.py          # Stacked LSTM model
│   ├── transformer_model.py   # Multi-head attention Transformer
│   ├── hybrid_model.py        # Hybrid LSTM + Transformer (NOVEL)
│   └── evaluation.py          # Metrics, plots, ISRO requirements
│
├── results/                   # Auto-generated plots and CSV
│
├── train.py                   # Full training (all 3 models)
├── demo.py                    # Quick demo (fast, fewer epochs)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Step 1 — Install Python 3.9+

Download from: https://www.python.org/downloads/

### Step 2 — Install VS Code

Download from: https://code.visualstudio.com/

### Step 3 — Open Project in VS Code

```
File → Open Folder → Select GNSS_Project/
```

### Step 4 — Open Terminal in VS Code

```
Terminal → New Terminal
```

### Step 5 — Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 6 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 7 — Generate Dataset

```bash
cd data
python generate_data.py
cd ..
```

### Step 8 — Run Quick Demo

```bash
python demo.py
```

### Step 9 — Run Full Training (For Research Paper Results)

```bash
python train.py
```

---

## Using Real ISRO Dataset

When you get the ISRO dataset:

1. Place the CSV file in `data/` folder
2. Open `train.py`
3. Change line 44:
   ```python
   DATA_PATH = 'data/YOUR_ISRO_FILE.csv'
   ```
4. Update `FEATURE_COLS` on line 52 to match your column names
5. Run `python train.py`

---

## Output Files (in results/)

| File | Description |
|------|-------------|
| `LSTM_predictions.png` | Actual vs Predicted (LSTM) |
| `Transformer_predictions.png` | Actual vs Predicted (Transformer) |
| `Hybrid_LSTM_Transformer_predictions.png` | Actual vs Predicted (Hybrid) |
| `*_error_distribution.png` | Error histogram + Q-Q plot (ISRO requirement) |
| `training_history.png` | Loss curves all models |
| `model_comparison.png` | RMSE bar chart |
| `metrics_table.csv` | Full metrics for research paper |
| `validity_period_rmse.csv` | RMSE at 15min/30min/1hr/2hr/24hr |

---

## ISRO Evaluation Requirements

| Validity Period | Implemented |
|----------------|-------------|
| 15 minutes | ✅ |
| 30 minutes | ✅ |
| 1 hour | ✅ |
| 2 hours | ✅ |
| 24 hours | ✅ |

**Error Distribution:** Shapiro-Wilk test checks normality. Closer to normal = better model.

---

## Research Contributions

1. **Multi-feature prediction:** Clock error + 3D ephemeris errors simultaneously
2. **Hybrid architecture:** Novel LSTM-Transformer fusion model
3. **Normal error distribution:** Evaluated with Shapiro-Wilk test
4. **Multi-horizon evaluation:** 15min to 24hr prediction validity

---

## Suggested Research Paper Title

> "Hybrid Transformer-LSTM Architecture for Time-Varying GNSS Satellite Clock and Ephemeris Error Prediction"

---

## Contact

For questions about the ISRO dataset: https://www.sac.gov.in/sih2025
