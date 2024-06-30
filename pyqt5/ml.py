import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data: midterm scores (X) and final scores (y)
data = {
    'Midterm_Score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'Final_Score': [58, 62, 66, 70, 74, 78, 82, 86, 90, 94]
}
df = pd.DataFrame(data)

# Separate the features (X) and target (y)
X = df[['Midterm_Score']].values
y = df['Final_Score'].values

# Manually split the data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    np.random.seed(42)  # For reproducibility
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Implement linear regression from scratch
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Add a column of ones to X to account for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal Equation: theta = (X_b.T * X_b)^(-1) * X_b.T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

# Create a SimpleLinearRegression instance
model = SimpleLinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Midterm Score')
plt.ylabel('Final Score')
plt.legend()
plt.show()

# Print the model coefficients
print(f'Coefficient (slope): {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

# Calculate and print the R^2 score
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

print(f'R^2 score: {r2_score(y_test, y_pred)}')
