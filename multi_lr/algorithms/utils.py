import numpy as np
class utils:
    @staticmethod
    def gen_query_set(n_features, n_samples = 100000, dtype = 'uniform', bounds = None):
        """
        Generate data
        """
        if dtype == 'uniform':
            return np.random.uniform(low=-1, high=1, size=(n_samples, n_features))
        elif dtype == 'uniform_int':
            return 2 * np.random.randint(low=0, high=2, size=(n_samples, n_features)) - 1
        elif dtype == 'norm':
            return np.random.randn(n_samples, n_features)
        elif dtype == 'data':
            min_x, max_x = bounds
            return np.random.uniform(low=min_x, high=max_x, size=(n_samples, n_features))
        else:
            raise ValueError('Unsupported data type')
    