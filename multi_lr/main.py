from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

from algorithms import MultiLR, MultiLRExtractor, MultiRLExtractor_local, utils

target = MultiLR.SoftmaxRegressionModel(method='multinomial')

