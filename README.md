# 🛡️ ThreatLens-AI

Classical ML-based network intrusion detection using the NSL-KDD dataset.

## Project Structure

```text
ThreatLens-AI/
├── utils.py          # Shared utilities, preprocessing, constants
├── train.py          # Training pipeline (RF + LR + Isolation Forest)
├── predict.py        # Prediction module + single-record test
├── app.py            # Streamlit dashboard
├── convert_test.py   # Converts NSL-KDD test data to prediction CSV
├── requirements.txt
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/           # Trained model files
└── plots/            # Evaluation plots
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Place the NSL-KDD dataset files in the `data/` directory:

- `KDDTrain+.txt`
- `KDDTest+.txt`

### 3. Train models

```bash
python train.py
```

This will:
- Load and preprocess the data
- Encode categorical features
- Scale numerical features
- Apply SMOTE for class balancing
- Train Random Forest and Logistic Regression
- Train Isolation Forest for anomaly detection
- Save trained models to `models/`
- Save evaluation plots to `plots/`

### 4. Prepare test data

```bash
python convert_test.py
```

This converts `KDDTest+.txt` into:

```text
data/test_traffic.csv
```

The generated CSV can be uploaded to the Streamlit dashboard for prediction.

### 5. Run predictions (CLI)

```bash
python predict.py
```

### 6. Launch Streamlit dashboard

```bash
streamlit run app.py
```

If you experience file-watcher issues on Windows, use:

```bash
streamlit run app.py --server.fileWatcherType none
```

## Models Used

| Model | Purpose | Output |
|-------|---------|--------|
| Random Forest | Classification | Attack category + probability |
| Logistic Regression | Classification (baseline) | Attack category + probability |
| Isolation Forest | Anomaly Detection | Anomaly score + label |

## Attack Categories

- **normal** — Benign traffic
- **dos** — Denial of Service
- **probe** — Network scanning/reconnaissance
- **r2l** — Remote to Local
- **u2r** — User to Root

## Risk Scoring

| Probability | Risk Level |
|-------------|------------|
| 0.0 – 0.3 | 🟢 Low |
| 0.3 – 0.7 | 🟡 Medium |
| 0.7 – 1.0 | 🔴 High |

## Dashboard

The Streamlit dashboard provides:

- Total records analyzed
- Attacks detected
- High-risk alerts
- Anomaly detection
- Attack distribution
- Risk distribution
- Attack probability
- Anomaly scores
- Detailed prediction results

## Data Processing

```text
NSL-KDD Dataset
       ↓
Data Preprocessing
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
SMOTE Class Balancing
       ↓
Machine Learning Models
       ↓
Attack Classification
       ↓
Risk & Anomaly Detection
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Streamlit
