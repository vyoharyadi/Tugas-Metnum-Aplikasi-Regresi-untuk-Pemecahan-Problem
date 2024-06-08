import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

data = pd.read_csv('Student_Performance.csv')

hours_studied = data['Hours Studied'].values
performance_index = data['Performance Index'].values

linear_model = LinearRegression()
linear_model.fit(hours_studied.reshape(-1, 1), performance_index)
linear_predictions = linear_model.predict(hours_studied.reshape(-1, 1))

def exponential_model(x, a, b):
    return a * np.exp(b * x)

params, _ = curve_fit(exponential_model, hours_studied, performance_index)
exponential_predictions = exponential_model(hours_studied, *params)

linear_error = np.sqrt(mean_squared_error(performance_index, linear_predictions))
exponential_error = np.sqrt(mean_squared_error(performance_index, exponential_predictions))

plt.figure(figsize=(10, 5))

plt.scatter(hours_studied, performance_index, color='blue', label='Data')
plt.plot(hours_studied, linear_predictions, color='red', label=f'Linear Model (RMS Error = {linear_error:.2f})')
plt.plot(hours_studied, exponential_predictions, color='green', label=f'Exponential Model (RMS Error = {exponential_error:.2f})')

plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.show()

print(f'RMS Error of Linear Model: {linear_error}')
print(f'RMS Error of Exponential Model: {exponential_error}')
