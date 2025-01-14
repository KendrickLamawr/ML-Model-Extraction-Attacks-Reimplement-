from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

from algorithms import MultiLR, MultiRLExtractor_local, utils

target = MultiLR.SoftmaxRegressionModel(method='multinomial')
target.train_iris()
# print(target.get_classes())
extractor = MultiRLExtractor_local.LocalMultiLRExtractor(target)
print(extractor.gen_query_set(3,1))
# print(utils.utils.gen_query_set(100,100).shape)
# print(extractor.get_num_features())

