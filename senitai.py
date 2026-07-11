# =============================================================================
# senitai.py - The AI Brain (Machine Learning Models)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def train_ai_models():
    """Trains the AI models and returns them."""
    np.random.seed(42)
    n_samples = 2000
    
    # 1. Generate Synthetic Data
    cpu_raw = np.clip(np.random.normal(40, 15, n_samples), 0, 100)
    ram_raw = np.clip(np.random.normal(50, 10, n_samples), 0, 100)
    disk_raw = np.clip(np.random.normal(25, 12, n_samples), 0, 100)
    net_raw = np.clip(np.random.normal(15, 8, n_samples), 0, 100)
    
    # Inject Anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples*0.08), replace=False)
    cpu_raw[anomaly_idx[:20]] = np.random.uniform(85, 99, 20)
    ram_raw[anomaly_idx[20:50]] += np.linspace(10, 40, 30)
    disk_raw[anomaly_idx[50:70]] = np.random.uniform(80, 99, 20)
    net_raw[anomaly_idx[70:]] = np.random.uniform(70, 99, len(anomaly_idx[70:]))
    
    df = pd.DataFrame({'cpu': cpu_raw, 'ram': np.clip(ram_raw,0,100), 'disk_io': disk_raw, 'network': net_raw})
    
    # 2. Feature Engineering
    for col in ['cpu', 'ram', 'disk_io', 'network']:
        df[f'{col}_roc'] = df[col].diff().fillna(0)
        df[f'{col}_std'] = df[col].rolling(5).std().fillna(0)
    df.fillna(0, inplace=True)
    
    # 3. Labels
    def assign_state(row):
        if row['cpu'] > 80 or row['ram'] > 85 or row['disk_io'] > 80: return 'Critical'
        if row['cpu'] > 60 or row['ram'] > 65 or row['disk_io'] > 55: return 'Stressed'
        return 'Healthy'
    df['true_state'] = df.apply(assign_state, axis=1)
    
    features = ['cpu', 'ram', 'disk_io', 'network', 'cpu_roc', 'ram_roc', 'cpu_std', 'ram_std']
    X = df[features]
    
    # 4. Train Models
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, df['true_state'])
    
    iso_f = IsolationForest(contamination=0.08, n_estimators=100, random_state=42)
    iso_f.fit(X)
    
    df['minutes_to_exhaust'] = np.where((df['cpu_roc'] > 0) & (df['cpu'] > 30), (100 - df['cpu'])/(df['cpu_roc']*60+0.001), 60)
    X_reg = df[['cpu', 'cpu_roc', 'ram', 'ram_roc']]
    y_reg = np.clip(df['minutes_to_exhaust'], 0, 60)
    lr = LinearRegression().fit(X_reg, y_reg)
    
    return rf, iso_f, lr, features