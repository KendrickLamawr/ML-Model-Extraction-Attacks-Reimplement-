import numpy as np
from sklearn.metrics import accuracy_score

class OfflineBase:
    def __init__(self, target, X_train, y_train, X_test, y_test, n_features):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.X_test = X_test

        self.n_features = n_features
        self.target = target
        self.clone = None
    
    def set_clone(self, clone):
        assert clone is not None, "Clone model cannot be None"
        if hasattr(clone, 'predict'):
            self.clone = clone.predict
        else:
            self.clone = clone

    def do(self):
        pass

    def benchmark(self):
        '''Evaluate cloned model'''
        assert self.clone is None, "Clone model cannot be None"

        #create 1000 observations with random features sampling from random distribution
        X_uniform = np.random.uniform(size=(1000, self.n_features)) 

        #use target model to predict X_uniform
        y_uniform_target = self.target(X_uniform) #self.target(X_uniform, count = False)
        #use clone model to predict X_uniform
        y_uniform_clone = self.clone(X_uniform)
        #calculate loss with uniform data (***Note: loss compared to target model, not true label)
        L_uniform = 1 - accuracy_score(y_uniform_target, y_uniform_clone)

        #===========
        y_test_target = self.target(self.X_test)
        y_test_clone = self.clone(self.X_test)
        #calculate loss with test data (***Note: loss compared to target model, not true label)
        L_test = 1 - accuracy_score(y_test_target, y_test_clone)

        #========== Calculate loss against true labels
        if -1 in self.y_test: #change negative labels to comply with true labels
            if -1 not in self.y_test_target:
                y_test_target = [y if y==1 else -1 for y in self.y_test_target]
            if -1 not in self.y_test_clone:
                y_test_clone = [y if y==1 else -1 for y in self.y_test_clone]
        print(self.__class__.__name__)
        print('Accuracy of target model: {}'.format(accuracy_score(y_test_target, self.y_test)))
        print('Accuracy of clone model: {}'.format(accuracy_score(y_test_clone, self.y_test)))
        print('Loss with uniform data: {}'.format(L_uniform))
        print('Loss with real test data: {}'.format(L_test))

        return L_uniform, L_test

