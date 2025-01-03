from abc import ABC, abstractmethod
class MultiLRExtractor:
    '''
    Extract coefficients from multiple logistic regression models.
    '''
    def __init__(self):
        self.classes = self.get_classes()
        self.X_train = None
    
    @abstractmethod
    def get_classes(self):
        return
    
    @abstractmethod
    def num_features(self):
        return
    
    def run_opti(self, loss, grad, X, Y, w_dim):