import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import unittest

# Load the data
df = pd.read_csv('Student_Performance.csv')

# Define the independent variable (Hours Studied) and dependent variable (Performance Index)
X = df['Hours Studied'].values.reshape(-1,1)
y = df['Performance Index'].values

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Exponential Regression
# Transform the dependent variable to apply Linear Regression
y_log = np.log(y)
exp_model = LinearRegression()
exp_model.fit(X, y_log)
y_pred_exp = np.exp(exp_model.predict(X))

# Calculate the Root Mean Squared Error (RMSE) for each model
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
rmse_exp = np.sqrt(mean_squared_error(y, y_pred_exp))

# Plot the data and the regression models
plt.figure(figsize=(12, 6))

# Plot the data
plt.scatter(X, y, color='blue', label='Data')

# Plot the Linear Regression model
plt.plot(X, y_pred_linear, color='red', label=f'Linear Regression (RMSE = {rmse_linear:.2f})')

# Plot the Exponential Regression model
plt.plot(X, y_pred_exp, color='green', label=f'Exponential Regression (RMSE = {rmse_exp:.2f})')

plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Regression Models for Student Performance')
plt.legend()
plt.show()

# Print the RMSE for each model
print(f'RMSE for Linear Regression: {rmse_linear}')
print(f'RMSE for Exponential Regression: {rmse_exp}')

# Define a function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Define a function to test the Linear Regression model
def test_linear_regression(X, y):
    linear_model.fit(X, y)
    y_pred = linear_model.predict(X)
    return np.allclose(y_pred, y_pred_linear, atol=1e-2)

# Define a function to test the Exponential Regression model
def test_exponential_regression(X, y):
    y_log = np.log(y)
    exp_model.fit(X, y_log)
    y_pred = np.exp(exp_model.predict(X))
    return np.allclose(y_pred, y_pred_exp, atol=1e-2)

# Test the calculate_rmse function
assert calculate_rmse(y, y_pred_linear) == rmse_linear
assert calculate_rmse(y, y_pred_exp) == rmse_exp

# Test the Linear Regression model
assert test_linear_regression(X, y)

# Test the Exponential Regression model
assert test_exponential_regression(X, y)
