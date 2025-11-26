"""Customer Churn Prediction Model"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

class ChurnPredictor:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = self._init_model()
    
    def _init_model(self):
        if self.model_type == 'xgboost':
            return XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05)
        return RandomForestClassifier(n_estimators=200, max_depth=15)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
        return self
