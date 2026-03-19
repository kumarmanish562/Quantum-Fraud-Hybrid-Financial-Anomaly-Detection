import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

class ClassicalModel:
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        self.params = params if params else {}
        self.model = None
        
    def build(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def train(self, X_train, y_train):
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise Exception("Model not trained yet.")
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return accuracy_score(y_test, y_pred)
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
