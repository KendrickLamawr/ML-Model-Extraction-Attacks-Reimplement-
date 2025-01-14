from algorithms.MultiLRExtractor import MultiLRExtractor
import numpy as np
class LocalMultiLRExtractor(MultiLRExtractor):
    def __init__(self, target, num_features=4):
        self.target = target
        MultiLRExtractor.__init__(self)
        self._w = None
        self.classes = self.get_classes()
        self.num_features = num_features
    
    def get_classes(self):
        return self.target.get_classes()
    
    def get_num_features(self):
        return self.num_features
    
    def set_target(self, target):
        self.target = target
    
    def multinomial_loss(W, X, y, lambda_reg):
        W = W.reshape(X.shape[1], -1)
        epsilon = 1e-6
        p_hat = super().softmax(X @ W)
        loss = -np.mean(np.sum(np.log(p_hat + epsilon) * y,axis=1)) + .5 * lambda_reg * np.sum(W**2)
        return loss
    def multinomial_grad(W, X, y, lambda_reg):
        W = W.reshape(X.shape[1], -1)
        cost = super().softmax(X @ W) - y
        gradient = 1/X.shape[0] * X.T @ cost + lambda_reg * W
        gradient = gradient.reshape(-1)
        return gradient

    def find_score(W, X):
        W.reshape(X.shape[1], -1)
        p_hat = super().softmax(X @ W)
        return np.argmax(p_hat, axis=1)
    
    # def run(self, 
    #         test_size=100000, random_seed=0,
    #         alphas=[0.5, 1, 2, 5, 10, 20, 50, 100],
    #         methods=["passive", "adapt-local", "adapt-oracle"],
    #         X_test = None, dataset_name = "Iris"):
    #     n_classes = len(self.get_classes())
        
    
    def run_opti(self, loss, grad, X, Y, w_dim):
        self.w, _ = super().run_opti(loss, grad, X, Y, w_dim)




    

