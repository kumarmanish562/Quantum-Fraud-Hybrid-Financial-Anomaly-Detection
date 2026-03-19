import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader
from .config import DATA_FILE_PATH, TEST_SIZE, RANDOM_SEED

class CreditCardDataset:
    def __init__(self, filepath=DATA_FILE_PATH):
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}. Please download the creditcard.csv from Kaggle.")
        
        self.df = pd.read_csv(self.filepath)
        
        # Basic preprocessing
        # Usually Time and Amount need scaling. 
        # For simplicity, let's assume standard fraud detection scaling.
        
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        return X, y

    def preprocess_and_split(self, use_smote=False):
        X, y = self.load_data()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        if use_smote:
            sm = SMOTE(random_state=RANDOM_SEED)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_torch_dataloaders(self, batch_size=32):
        if self.X_train is None:
            self.preprocess_and_split(use_smote=True) # Usually good for Neural Nets on imbalanced data
            
        train_dataset = TorchDataset(self.X_train, self.y_train)
        test_dataset = TorchDataset(self.X_test, self.y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
