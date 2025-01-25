from abc import ABC, abstractmethod
from algorithms import utils
import numpy as np
from scipy.optimize import minimize
from algorithms.utils import utils
import pandas as pd
from algorithms.MultiLR import SoftmaxRegressionModel
from sklearn.preprocessing import OneHotEncoder

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def multinomial_loss(W, X, y, lambda_reg):
    W = W.reshape(X.shape[1], -1)
    epsilon = 1e-6
    p_hat = softmax(X @ W)
    loss = -np.mean(np.sum(np.log(p_hat + epsilon) * y,axis=1)) + .5 * lambda_reg * np.sum(W**2)
    return loss

def multinomial_grad(W, X, y, lambda_reg):
    W = W.reshape(X.shape[1], -1)
    cost = softmax(X @ W) - y
    gradient = 1/X.shape[0] * X.T @ cost + lambda_reg * W
    gradient = gradient.reshape(-1)
    return gradient



class MultiLRExtractor:
    '''
    Extract coefficients from multiple logistic regression models.
    '''
    def __init__(self, target, num_features=4):
        self.target = None
        self.best_w = None
        self.classes = self.get_classes()
        self.num_features = num_features
        self.X_train = None
    
    def one_hot(self, X):
        categorical_idx = [i for i in range(X.shape[1])]
        encoder = OneHotEncoder(categorical_features=categorical_idx, sparse=False)
        return encoder.fit_transform(pd.DataFrame(X))

    def get_classes(self):
        return self.num_features
    
    def num_features(self):
        return
    
    def gen_query_set(self, n_features, n_samples):
        return utils.gen_query_set(n_features, n_samples)

    def set_target(self, model):
        self.target = model
    
    def predict_probas(self, X, w):
        return softmax(X @ w)
    def query_probas(self, X, w):
        X_transformed = self.one_hot(X)
        p_hat = self.predict_probas(X_transformed, w)
        return p_hat

    def find_score(self, W, X):
        W.reshape(X.shape[1], -1)
        p_hat = softmax(X @ W)
        return np.argmax(p_hat, axis=1)
    
    def run_opti(self, loss, grad, X, Y):
        '''
        Optimization procedure
        '''
        n_features = X.shape[1]
        n_classes = Y.shape[1]
        best_w = None
        best_acc = 0
        
        n_inters = 5
        alphas = [10**x for x in range(-20,4)]

        fprimes = [grad]
        
        for fprime in fprimes:
            for alpha in alphas:
                w0 = 1e-8 * np.random.randn(n_features * n_classes)

                # num_unknows = len(w0.ravel())
                method = "L-BFGS-B"
                optimLogitBFGS = minimize(loss, x0=w0,
                                        method = method,
                                        args = (X, Y, alpha),
                                        jac = fprime,
                                        options={'gtol': 1e-6,
                                                'disp': True,
                                                'maxiter': 100})
                wopt = optimLogitBFGS.x
                wopt_reshape = wopt.reshape(n_features, n_classes)
                acc = self.compare_models(X, wopt_reshape)
                print(f"f'={fprime}, alpha={alpha}:")
                print(f'Clone model predictions are {acc}% similar to target model predictions.')
                if acc > best_acc:
                    best_w = wopt_reshape
                    best_acc = acc
        return best_w, best_acc
    
    # def find_coefficients(self, budget, adapt = False):

    def compare_with_target(self, X, wopt):
        clone_pred = np.argmax(softmax(X @ wopt), axis = 1)
        
        target_pred = self.target.predict(X)
        acc = (np.mean(clone_pred == target_pred))
        
        return acc
    def baseline_model(self, X):
        Y = pd.Series(self.target.predict(X))
        model = SoftmaxRegressionModel('multinomial')
        model.fit(X, Y)
        return model
    

    
    def find_coeffs(self, m, baseline = False, adapt=False):
        n_classes = len(self.classes)
        n_features = self.num_features()

        #generate random queries
        if not adapt:
            X = self.gen_query_set(n_features=n_features, n_samples = m)
        else: 
            X = utils.line_search_oracle(n_features, m, self.query, self.gen_query_set)
        self.X_train = X

        # get the probabilities for all queries
        if baseline:
            model = self.baseline_model(X)
            return model 
        Y = self.query_probas(X)

        return self.select_and_run_opti(X,Y)
    
    def find_coeffs_adaptive(self, step, query_budget, baseline=False):
        assert query_budget > 0
        n_classes = len(self.classes)
        n_features = self.num_features()

        X = self.gen_query_set
    
    def select_and_run_opti(self, X, Y):
        best_w, best_acc = self.run_opti(multinomial_loss,
                                         multinomial_grad,
                                         X, Y)
        return best_w, best_acc
        
    
    
    


            
