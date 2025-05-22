# Credit Card Fraud Detection ML Pipeline

A Python-based end-to-end machine learning pipeline for detecting fraudulent credit card transactions on a highly imbalanced dataset (0.17% fraud).

## Features

* **Data Ingestion**: Reads CSV of 280K transactions (PCA features V1–V28, `Amount`, `Class`).
* **Preprocessing Pipeline**: StandardScaler + SMOTE oversampling inside an `imblearn` Pipeline to handle class imbalance without test leakage.
* **Modeling**: Supports Random Forest and XGBoost, with hyperparameter tuning via `RandomizedSearchCV` optimizing PR-AUC.
* **Probability Calibration**: Optional isotonic calibration for better probability estimates.
* **Threshold Tuning**: Automatically selects probability thresholds to achieve a target recall (e.g., 90%).
* **CLI Tool**: Command-line interface with flags for data path, model choice, tuning, calibration, recall target, threshold, and artifact saving.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/creditcard-fraud-detection.git
   cd creditcard-fraud-detection
   ```
2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib

   ```

## Usage

```bash
python fraud_detection.py \
  --data_path path/to/creditcard.csv \
  --model rf \
  --tune \
  --calibrate \
  --recall 0.9 \
  --threshold 0.5 \
  --save
```

* `--data_path`  Path to the CSV dataset
* `--model`      `rf` (Random Forest) or `xgb` (XGBoost)
* `--tune`       Enable hyperparameter tuning (PR-AUC optimization)
* `--calibrate`  Calibrate probabilities with isotonic regression
* `--recall`     Target recall (0–1) for threshold tuning
* `--threshold`  Fixed probability threshold (if omitted, auto-tuned)
* `--save`       Serialize the trained model to `best_model.pkl`



## Results

On held-out test set (0.17% fraud):

* **ROC-AUC**: \~0.92
* **PR-AUC**: \~0.88
* **Recall @ 90%**: achieves target recall with tunable precision

## Contributing

Contributions welcome! Please submit issues or pull requests for bug fixes, new features, or improvements.

