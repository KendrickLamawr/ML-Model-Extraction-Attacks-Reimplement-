from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
# Prepare the data
X = iris.data  # Feature matrix
y = iris.target  # Target vector

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use softmax regression (multi-class logistic regression)
target_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
target_model.fit(X_train, y_train)

# Predict on the test set
y_pred = target_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

accuracy, y_pred[:5]  # Show accuracy and first 5 predictions
print(accuracy)

