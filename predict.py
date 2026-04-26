import pandas as pd
import numpy as np
from utils import load_artifacts, get_risk_level, CATEGORICAL_COLS


def load_models():
    artifacts = load_artifacts([
        'random_forest', 'logistic_regression',
        'isolation_forest', 'encoders', 'scaler', 'feature_names'
    ])
    return artifacts


def preprocess_input(df, encoders, scaler, feature_names):
    df = df.copy()
    
    # Drop columns not needed
    for col in ['label', 'attack_category', 'is_attack', 'difficulty']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    
    X_scaled = scaler.transform(df)
    return X_scaled


def predict(input_df, model_choice='random_forest'):
    artifacts = load_models()
    
    rf = artifacts['random_forest']
    lr = artifacts['logistic_regression']
    iso = artifacts['isolation_forest']
    encoders = artifacts['encoders']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    
    model = rf if model_choice == 'random_forest' else lr
    
    X = preprocess_input(input_df, encoders, scaler, feature_names)
    
    # Classification predictions
    attack_type = model.predict(X)
    proba = model.predict_proba(X)
    
    # Max probability for the predicted class
    max_proba = np.max(proba, axis=1)
    
    # Attack probability (probability of not being normal)
    classes = list(model.classes_)
    if 'normal' in classes:
        normal_idx = classes.index('normal')
        attack_prob = 1 - proba[:, normal_idx]
    else:
        attack_prob = max_proba
    
    risk_levels = [get_risk_level(p) for p in attack_prob]
    
    # Anomaly detection
    anomaly_scores = iso.decision_function(X)
    anomaly_labels = iso.predict(X)  # -1 = anomaly, 1 = normal
    
    results = pd.DataFrame({
        'predicted_category': attack_type,
        'is_attack': (attack_type != 'normal').astype(int),
        'attack_probability': np.round(attack_prob, 4),
        'risk_level': risk_levels,
        'anomaly_score': np.round(anomaly_scores, 4),
        'is_anomaly': (anomaly_labels == -1).astype(int)
    })
    
    return results


def predict_single(record: dict, model_choice='random_forest'):
    df = pd.DataFrame([record])
    results = predict(df, model_choice)
    result = results.iloc[0]
    
    print("\n=== Prediction Result ===")
    print(f"  Attack Category : {result['predicted_category']}")
    print(f"  Is Attack       : {'Yes' if result['is_attack'] else 'No'}")
    print(f"  Attack Prob.    : {result['attack_probability']:.2%}")
    print(f"  Risk Level      : {result['risk_level']}")
    print(f"  Anomaly Score   : {result['anomaly_score']:.4f}")
    print(f"  Is Anomaly      : {'Yes' if result['is_anomaly'] else 'No'}")
    
    return result


# Sample input matching NSL-KDD format
SAMPLE_NORMAL = {
    'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
    'src_bytes': 215, 'dst_bytes': 45076, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
    'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
    'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
    'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
    'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 1,
    'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
    'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
    'srv_diff_host_rate': 0.0, 'dst_host_count': 0, 'dst_host_srv_count': 0,
    'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
}

SAMPLE_ATTACK = {
    'duration': 0, 'protocol_type': 'icmp', 'service': 'ecr_i', 'flag': 'SF',
    'src_bytes': 1032, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
    'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
    'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
    'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
    'is_host_login': 0, 'is_guest_login': 0, 'count': 511, 'srv_count': 511,
    'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
    'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
    'srv_diff_host_rate': 0.0, 'dst_host_count': 255, 'dst_host_srv_count': 255,
    'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
}


if __name__ == '__main__':
    print("--- Testing Normal Traffic ---")
    predict_single(SAMPLE_NORMAL)
    
    print("\n--- Testing Attack Traffic ---")
    predict_single(SAMPLE_ATTACK)
