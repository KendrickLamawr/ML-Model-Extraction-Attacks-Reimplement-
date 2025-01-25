from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

from algorithms import MultiLR, MultiLRExtractor

target = MultiLR.SoftmaxRegressionModel(method='multinomial')
target.train_iris()
# print(target.get_classes())
extractor = MultiLRExtractor.MultiLRExtractor(target)
print(extractor.gen_query_set(3,1))
# print(utils.utils.gen_query_set(100,100).shape)
# print(extractor.get_num_features())

