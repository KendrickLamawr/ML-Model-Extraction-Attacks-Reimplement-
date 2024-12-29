import numpy as np
class LRSolver:
    def __init__(self):
        self.w = None
    
    def solve(self, X_test, y_test):
        # print(np.ones((X_test.shape[0], 1)).shape, X_test.shape)
        X_design = np.hstack((np.ones((X_test.shape[0], 1)), X_test)) #design matrix, has the size of [number of observations, number of features + 1]
        # print(X_test.shape, X_design[:, 1:].shape)
        # X_design[:, 1:] = X_test
        # print(X_design.shape, self.sigmoid_inverse(y_test).shape)
        #solve least squares problem to find weights and bias
        self.w, residuals, rank, s = np.linalg.lstsq(X_design, self.sigmoid_inverse(y_test), rcond=None)

    def sigmoid_inverse(self, probabilities):

        if np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("All probabilities must be in the range (0, 1).")
        
        return -np.log(-1+1/probabilities)
    
    def predict(self, X_test):
        X_design = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        # print(np.dot(X_design, self.w).shape, X_design.shape, self.w.shape)
        probs = 1/(1+np.exp(-np.dot(X_design, self.w)))
        # print(probs)
        return (probs >= 0.5).astype(int)
    
    def predict_proba(self, X_test):
        X_design = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        probs = 1/(1+np.exp(-np.dot(X_design, self.w)))
        return probs

    