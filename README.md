# 🛡️ Cyber Threat Detection System
Classical ML-based network intrusion detection using NSL-KDD dataset.

## Project Structure
```
cyber_threat_detection/
├── utils.py          # Shared utilities, preprocessing, constants
├── train.py          # Training pipeline (RF + LR + Isolation Forest)
├── predict.py        # Prediction module + single-record test
├── app.py            # Streamlit dashboard
├── requirements.txt
├── data/             # Place NSL-KDD dataset here
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/           # Auto-created after training
└── plots/            # Auto-created after training
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NSL-KDD Dataset
Download from: https://www.unb.ca/cic/datasets/nsl.html
- `KDDTrain+.txt`
- `KDDTest+.txt`

Place both files in the `data/` directory.

### 3. Train models
```bash
python train.py
```
This will:
- Load and preprocess data
- Apply SMOTE for class balancing
- Train Random Forest + Logistic Regression
- Train Isolation Forest for anomaly detection
- Save models to `models/`
- Save evaluation plots to `plots/`

### 4. Run predictions (CLI)
```bash
python predict.py
```

### 5. Launch Streamlit dashboard
```bash
streamlit run app.py
```

## Sample Input Format
41 NSL-KDD features in CSV (no header):
```
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.0,...
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
- **probe** — Surveillance/Scanning
- **r2l** — Remote to Local
- **u2r** — User to Root

## Risk Scoring
| Probability | Risk Level |
|-------------|-----------|
| 0.0 – 0.3  | 🟢 Low     |
| 0.3 – 0.7  | 🟡 Medium  |
| 0.7 – 1.0  | 🔴 High    |
