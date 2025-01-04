from abc import ABC, abstractmethod
from algorithms import utils
import numpy as np
from scipy.optimize import minimize
class MultiLRExtractor:
    '''
    Extract coefficients from multiple logistic regression models.
    '''
    def __init__(self):
        # self.classes = self.get_classes()
        # self.X_train = None
        self.target = None
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # @abstractmethod
    # def get_classes(self):
    #     return
    
    # @abstractmethod
    # def num_features(self):
    #     return
    
    def gen_query_set(self, n_features, n_samples):
        return gen_query_set(n_features, n_samples)

    def set_target(self, model):
        self.target = model
    
    def run_opti(self, loss, grad, X, Y, w_dim=None):
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
    
    def compare_with_target(self, X, wopt):
        clone_pred = np.argmax(self.softmax(X @ wopt), axis = 1)
        
        target_pred = self.target.predict(X)
        acc = (np.mean(clone_pred == target_pred))
        
        return acc