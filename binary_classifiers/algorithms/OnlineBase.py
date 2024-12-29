import numpy as np
import sys

from utils.logger import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class OnlineBase:
    '''Class for interacting with online models'''
    def __init__(self, name, pos_label, neg_label, model, n_features, ftype, error_threshold):
        '''name: model name
        pos_label: positive label
        neg_label: negative label
        model: model object
        n_features: number of features
        ftype: feature type (continuous, categorical, ordinal)
        error_threshold: #'''
        self.name = name
        self.pos_label, self.neg_label = pos_label, neg_label
        self.model = model

        #number of consumed queries
        self.q = 0 

        self.error_threshold = error_threshold
        self.n_features = n_features
        self.ftype = ftype

        #budget: maximum number of queries allowed
        self.budget = -1 #-1 means unlimited budget

        #points near decision boundary, means certainty/confidence is low for these data points
        self.pts_near_b = []
        self.pts_near_b_labels = []

    def set_budget(self, budget):
        '''Set the budget for the model'''
        self.budget = budget

    def add_budget(self, b):
        '''Add a certain amount of budget'''
        self.budget += b if self.budget != -1 else b

    def random_one_point(self, size, label=None, distribution=None):
        '''Generate random data with given size and label'''
        if distribution is not None:
            self.ftype = distribution.type
            mean = distribution.mean
            low, high = distribution.low, distribution.high
        else:
            mean = [0]*size
            low, high = -1, 1
        
        #check if label is a valid label
        if label is not None:
            assert label in [self.pos_label, self.neg_label], 'Unknown label'
        
        def random_binary_data(size):
            return 2*np.random.randint(0, 2, size) - 1
        def random_normal_data(size):
            if distribution is not None and distribution.type == 'norm':
                assert len(distribution.mean) == self.n_features, 'Mean array expects to match the number of features'
                return np.random.normal(loc = distribution.mean)
            assert len(mean) == self.n_features
            return np.random.normal(loc = mean)
        def random_uniform_data(size):
            return np.random.uniform(low, high, size)
        
        assert self.ftype in ['binary', 'uniform', 'norm'], 'Unknown feature type'
        if self.ftype == 'binary':
            create_data = random_binary_data
        elif self.ftype == 'uniform':
            create_data = random_uniform_data
        elif self.ftype == 'norm':
            create_data = random_normal_data
        
        if label:
            point = create_data(size)
            pred = self.query(point)
            if pred == label:
                return point
            else:
                logger.debug('Want %d got %d', label, pred)
        else:
            return create_data(size)
    
    def query(self, x, count=True):
        '''predict the label of the given data point using self.model (this class is for target model)'''
        if count:
            self.q += 1
            logger.debug('Query %d: %s', self.q, x)
            if self.q > 0 and self.q % 100 == 0:
                logger.debug("{} queries consumed".format(self.q))
            if self.budget!= -1 and self.q > self.budget:
                raise RunOutOfBudgetError
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            return self.model(x)
    
    # def push_pair_to_boundary(self, x_neg, x_pos, max_distance):
    #     '''
    #     Pull a pair of data points close to the boundary
    #     x_neg: negative data point
    #     x_pos: positive data point
    #     '''
    #     assert self.query(x_neg, count = False) == self.neg_label
    #     assert self.query(x_pos, count = False) == self.pos_label


class RunOutOfBudgetError(Exception):
    pass


#=====Distribution class=====
class Distribution(object):
    def __init__(self, type, range, mean):
        self.type = type
        self.range = range
        self.mean = mean



