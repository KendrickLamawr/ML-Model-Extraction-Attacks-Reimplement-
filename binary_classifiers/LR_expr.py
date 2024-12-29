from algorithms.LogisticRegression import LogisticModel
from algorithms.LogisticRegressionSolver import LRSolver

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
'''
Experiments on extracting logistic regression model using the adult income dataset
https://www.kaggle.com/code/terrycheng/adult-income-dataset-analysis/input 
'''
#  initialize target model
model = LogisticModel(max_iter = 100000)
data, labels = load_svmlight_file("binary_classifiers/targets/adult/train.scale")
model.train(data, labels)


X_test, y_test = load_svmlight_file("binary_classifiers/targets/adult/test")
num_features = X_test.shape[1]
# choose 109 random rows corresponding to weights of 108 features + 1 bias
indices = np.random.choice(X_test.shape[0], 109, replace=False)
X_solve = X_test[indices].toarray()  # convert sparse matrix to dense matrix for easier manipulation
y_solve = model.predict(X_solve)


# extract model parameters
clone_model = LRSolver()
clone_model.solve(X_solve, y_solve)
# print(accuracy_score(y_test, clone_model.predict(X_test.toarray())))
pred_clone = clone_model.predict(X_test.toarray())
pred_oracle = model.predict(X_test.toarray())
'''COMPARE ACCURACY OF CLONE AND ORACLE
We only need 109 data points to recover 99.57% a model which was trained on 34190 data points.
'''
print(accuracy_score(np.argmax(pred_clone, axis=1), np.argmax(pred_oracle, axis=1)))
