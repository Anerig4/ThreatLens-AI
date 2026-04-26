import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# NSL-KDD column names
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Attack category mapping
ATTACK_CATEGORY = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos',
    'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'sendmail': 'r2l', 'named': 'r2l', 'snmpattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}


def load_data(train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)
    return train_df, test_df


def preprocess(df, fit_encoders=None, fit_scaler=None):
    df = df.copy()
    
    # Drop difficulty column
    if 'difficulty' in df.columns:
        df.drop('difficulty', axis=1, inplace=True)
    
    # Map labels to attack category
    df['label'] = df['label'].str.strip('.')
    df['attack_category'] = df['label'].map(lambda x: ATTACK_CATEGORY.get(x, 'unknown'))
    df['is_attack'] = (df['attack_category'] != 'normal').astype(int)
    
    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        if fit_encoders is None:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = fit_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Numerical features
    feature_cols = [c for c in df.columns if c not in ['label', 'attack_category', 'is_attack']]
    X = df[feature_cols].copy()
    
    # Scale
    if fit_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X_scaled = scaler.transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, df['is_attack'], df['attack_category'], encoders if fit_encoders is None else fit_encoders, scaler if fit_scaler is None else fit_scaler, feature_cols


def get_risk_level(prob):
    if prob < 0.3:
        return 'Low'
    elif prob < 0.7:
        return 'Medium'
    else:
        return 'High'


def save_artifacts(models_dict, path='models/'):
    os.makedirs(path, exist_ok=True)
    for name, obj in models_dict.items():
        with open(os.path.join(path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f)
    print(f"Saved {len(models_dict)} artifacts to {path}")


def load_artifacts(names, path='models/'):
    artifacts = {}
    for name in names:
        fpath = os.path.join(path, f'{name}.pkl')
        with open(fpath, 'rb') as f:
            artifacts[name] = pickle.load(f)
    return artifacts
