from algorithms.MultiLRExtractor import MultiLRExtractor
class LocalMultiLRExtractor(MultiLRExtractor):
    def __init__(self):
        self.model = None
        MultiLRExtractor.__init__(self)
        self._w = None
    
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
    
    
    def run_opti(self, loss, grad, X, Y, w_dim):
        self.w, _ = super().run_opti(loss, grad, X, Y, w_dim)




    

