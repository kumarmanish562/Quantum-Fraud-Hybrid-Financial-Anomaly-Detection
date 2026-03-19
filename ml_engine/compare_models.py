import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config but we might override data loading
try:
    from ml_engine.config import HQNN_PARAMS, XGB_PARAMS, RANDOM_SEED, DATA_FILE_PATH
    from ml_engine.models.hybrid_nn import HybridQNN
    from ml_engine.dataset import CreditCardDataset
except ImportError:
    # If running from inside ml_engine without package structure
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import HQNN_PARAMS, XGB_PARAMS, RANDOM_SEED, DATA_FILE_PATH
    from models.hybrid_nn import HybridQNN
    from dataset import CreditCardDataset

def load_sampled_data(fraction=0.1):
    print(f"Loading data from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)
    
    # Sample data for speed
    print(f"Sampling {fraction*100}% of data for quick check...")
    df_sample = df.sample(frac=fraction, random_state=RANDOM_SEED)
    
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']
    
    print(f"Sample size: {len(X)}")
    print(f"Class distribution: \n{y.value_counts()}")
    
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

def train_eval_xgboost(X_train, X_test, y_train, y_test):
    print("\n--- Training XGBoost ---")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.4f}")
    return acc

def train_eval_rf(X_train, X_test, y_train, y_test):
    print("\n--- Training Random Forest ---")
    model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED) # Reduced estimators for speed
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
    return acc

def train_eval_hybrid(X_train, X_test, y_train, y_test):
    print("\n--- Training Hybrid QNN (Lite) ---")
    
    # Scale for NN
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Convert to Torch
    X_train_tensor = torch.FloatTensor(X_train_sc)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_sc)
    
    # Subset for QNN training (very slow otherwise)
    # Take only first 200 samples for training demo
    limit = 200
    print(f"Training QNN on subset of {limit} samples because simulation is slow...")
    X_train_sub = X_train_tensor[:limit]
    y_train_sub = y_train_tensor[:limit]
    
    n_features = X_train.shape[1]
    model = HybridQNN(n_features, HQNN_PARAMS['n_qubits'], HQNN_PARAMS['n_layers'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=HQNN_PARAMS['learning_rate'])
    
    epochs = 2
    batch_size = 16
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_sub.size()[0])
        total_loss = 0
        batches = 0
        
        for i in range(0, X_train_sub.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_sub[indices], y_train_sub[indices]
            
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/batches:.4f}")

    model.eval()
    with torch.no_grad():
        # Evaluate on full test set (fast enough) or subset
        # Evaluating QNN on 1000s of samples is also slow, let's limit eval
        eval_limit = 500
        outputs = model(X_test_tensor[:eval_limit])
        predicted = (outputs > 0.5).float().flatten()
        y_test_sub = torch.FloatTensor(y_test.values)[:eval_limit]
        
        acc = (predicted == y_test_sub).float().mean().item()
        
    print(f"Hybrid QNN Accuracy (on partial test set): {acc:.4f}")
    return acc

def main():
    try:
        # Load 10% of data
        X_train, X_test, y_train, y_test = load_sampled_data(fraction=0.1)
    except Exception as e:
        print(e)
        return

    acc_xgb = train_eval_xgboost(X_train, X_test, y_train, y_test)
    acc_rf = train_eval_rf(X_train, X_test, y_train, y_test)
    acc_hybrid = train_eval_hybrid(X_train, X_test, y_train, y_test)
    
    print("\n=== Model Comparison Results ===")
    print(f"XGBoost Accuracy:       {acc_xgb:.2%}")
    print(f"Random Forest Accuracy: {acc_rf:.2%}")
    print(f"Hybrid QNN Accuracy:    {acc_hybrid:.2%}")
    
    if acc_xgb > 0.96 or acc_rf > 0.96:
        print("\n✅ Requirement Met: Model accuracy is above 96%.")
    else:
        print("\n⚠️ Requirement Not Met.")

if __name__ == "__main__":
    main()
