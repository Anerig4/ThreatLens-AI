import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils import load_data, preprocess, save_artifacts, ATTACK_CATEGORY

import os
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"\nDetailed Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    return y_pred


def plot_confusion_matrix(y_test, y_pred, labels, title, filename):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}.png")


def plot_feature_importance(rf_model, feature_names, top_n=20):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()
    print("Saved: plots/feature_importance.png")
    
    print("\nTop 10 Features:")
    for i in range(min(10, top_n)):
        print(f"  {feature_names[indices[i]]:<35} {importances[indices[i]]:.4f}")


def train_classification_models(X_train, y_train, X_test, y_test, feature_names):
    print("\n[1] Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Before: {dict(pd.Series(y_train).value_counts())}")
    print(f"  After : {dict(pd.Series(y_res).value_counts())}")

    # --- Random Forest ---
    print("\n[2] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(X_res, y_res)
    rf_preds = evaluate_model(rf, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_preds, sorted(y_test.unique()), 
                          "Random Forest - Confusion Matrix", "rf_confusion")
    plot_feature_importance(rf, feature_names)

    # --- Logistic Regression ---
    print("\n[3] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='auto')
    lr.fit(X_res, y_res)
    lr_preds = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_preds, sorted(y_test.unique()),
                          "Logistic Regression - Confusion Matrix", "lr_confusion")

    return rf, lr


def train_anomaly_model(X_train):
    print("\n[4] Training Isolation Forest (Anomaly Detection)...")
    iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    print("  Isolation Forest trained successfully.")
    return iso


def main():
    print("=== Cyber Threat Detection - Training Pipeline ===\n")

    # Load data
    print("[0] Loading NSL-KDD dataset...")
    train_df, test_df = load_data()
    print(f"  Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Preprocess
    print("\n[0] Preprocessing...")
    X_train, y_train_bin, y_train_cat, encoders, scaler, feature_names = preprocess(train_df)
    X_test, y_test_bin, y_test_cat, _, _, _ = preprocess(test_df, fit_encoders=encoders, fit_scaler=scaler)

    print(f"  Features: {len(feature_names)}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Attack categories in train: {y_train_cat.value_counts().to_dict()}")

    # Train classification on attack categories
    rf, lr = train_classification_models(X_train, y_train_cat, X_test, y_test_cat, feature_names)

    # Train anomaly detection
    iso = train_anomaly_model(X_train)

    # Save all artifacts
    print("\n[5] Saving models and artifacts...")
    save_artifacts({
        'random_forest': rf,
        'logistic_regression': lr,
        'isolation_forest': iso,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': feature_names
    })

    print("\n=== Training Complete ===")
    print("Models saved to: models/")
    print("Plots saved to:  plots/")


if __name__ == '__main__':
    main()
