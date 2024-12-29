from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
class LogisticModel:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict_proba(X_test)
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)[:, 1]
        return accuracy_score(y_test, y_pred)

    