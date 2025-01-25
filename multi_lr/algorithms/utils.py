import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
import math
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
        
    @staticmethod
    def query_count(X, Y, eps):
        dist = squareform(pdist(X, 'euclidean'))
        tot = 0
        for (i,j) in utils.all_pairs(Y):
            if dist[i][j] > eps:
                tot += math.ceil(np.log2(dist[i][j]/eps))
        return tot

    @staticmethod
    def line_search_oracle(n_features, budget, oracle, query_generator, error_thres=1e-1):
        """
        Line search oracle for finding the optimal query point
        """
        X_init = query_generator(n_features,1)
        Y = oracle.predict(X_init)

        budget_0 = budget
        budget -= 1

        step = (budget+3)/4
        while utils.query_count(X_init, Y, error_thres) <= budget:
            x = query_generator(n_features, step)
            y = oracle.predict(x)
            X_init = np.vstack((X_init, x))
            Y = np.hstack((Y, y))
            budget -= step
        
        if budget <= 0:
            assert len(X_init) >= budget_0
            return X_init[0:budget_0]
        
        Y = Y.flatten()
        idx1, idx2 = zip(*utils.all_pairs(Y))
        idx1 = list(idx1)
        idx2 = list(idx2)
        samples = utils._line_search(X_init, Y, idx1, idx2, oracle.predict, error_thres, append = True)
        assert len(samples) >= budget_0
        return samples[0:budget_0]
    
    @staticmethod
    def _line_search(X, Y, idx1, idx2, predict_fn, error_thres, append=False):
        v1 = X[idx1,:] 
        y1 = Y[idx1]
        v2 = X[idx2, :]
        y2 = Y[idx2]

        assert np.all(y1!=y2)
        if append:
            samples = X

        while np.any(np.sum((v1-v2)**2, axis=-1)**(1./2) > error_thres):
            mid = 0.5 * (v1 + v2)
            y_mid = predict_fn(mid)

            index1 = np.where(y_mid != y1)[0]
            index2 = np.where(y_mid == y2)[0]

            if len(index1):
                v2[index1,:] = mid[index1, :]
            if len(index2):
                v1[index2,:] = mid[index2,:]
            
            if append:
                samples = np.vstack((samples, mid))
        if append:
            return samples
        else:
            return np.vstack((v1,v2))


    @staticmethod
    def query_count(X, Y, error_thres):
        dist = squareform(pdist(X, 'euclidean')) #pairwise distances
        res = 0
        
        for (i,j) in utils.all_pairs(Y):
            if dist[i,j] > error_thres:
                res += math.ceil(np.log2(dist[i][j]/error_thres))
        return res

    @staticmethod
    def all_pairs(Y):
        '''
        Return all pairs with different labels
        '''
        classes = pd.Series(Y).unique().tolist()
        return [(i, j)
                for i in range(len(Y))              
                for c in classes                    
                if c != Y[i]
                for j in np.where(Y == c)[0][0:1]   
                if i > j]
    
 



    