import numpy as np
def gen_query_set(n, test_size = 100000, dtype = 'uniform', bounds = None):
    """
    Generate data
    """
    if dtype == 'uniform':
        X = np.random.uniform(low=-1, high=1, size=(test_size, n))
    elif dtype == 'uniform_int':
        return 2 * np.random.randint(low=0, high=2, size=(test_size, n)) - 1
    elif dtype == 'norm':
        return np.random.randn(test_size, n)
    elif dtype == 'data':
        min_x, max_x = bounds
        return np.random.uniform(low=min_x, high=max_x, size=(test_size, n))
    else:
        raise ValueError('Unsupported data type')
    