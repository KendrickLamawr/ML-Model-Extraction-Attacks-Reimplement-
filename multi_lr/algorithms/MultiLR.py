from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.datasets import load_iris
class SoftmaxRegressionModel:
    def __init__(self, method, max_iter = 200):
        if method == 'multinomial':
            self.model = LogisticRegression(multi_class="multinomial", 
                                            solver="lbfgs", 
                                            max_iter=max_iter)
        else:
            self.model = LogisticRegression(multi_class="ovr")
    def train_iris(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        self.train(X, y)
    def get_classes(self):
        return self.model.classes_
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