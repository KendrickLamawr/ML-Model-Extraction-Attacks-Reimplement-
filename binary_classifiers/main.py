from algorithms.OnlineBase import *
base = OnlineBase('name', 'pos_label', 'neg_label', 'model', 10, 'norm', 0.01)
print(base.random_one_point(10))
import numpy as np

# Example matrix X_test with shape (100, 108)
X_test = np.random.rand(109, 108)

# Create a column of ones with shape (100, 1) and horizontally stack it with X_test
ones_column = np.ones((X_test.shape[0], 1))  # Create a column vector of ones with 100 rows
design_matrix = np.hstack((ones_column, X_test))  # Horizontally stack the ones column with X_test

# The resulting design_matrix will have shape (100, 109)
print(design_matrix.shape)  # Should print (100, 109)
print(ones_column.shape, X_test.shape)