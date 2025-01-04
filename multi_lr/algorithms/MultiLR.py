from sklearn.linear_model import LogisticRegression
import pickle
class SoftmaxRegressionModel:
    def __init__(self, method):
        if method == 'multinomial':
            self.model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        else:
            self.model = LogisticRegression(multi_class="ovr")
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def predict(self, X_test):
        return self.model.predict(X_test)