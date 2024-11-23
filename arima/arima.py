import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load dataset
# 
# timestamp
# user_count 




# data = pd.read_csv('stock_data_with_demand.csv', index_col='date', parse_dates=True, date_format='%Y-%m-%d')
# data = pd.read_csv('data2.csv', index_col='timestamp', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S.%f')
data = pd.read_csv('data2_with_demand.csv', index_col='timestamp', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S.%f')
# data = pd.read_csv('data_generated.csv', index_col='date', parse_dates=True, date_format='%Y-%m-%d')
# data = pd.read_csv('data2_interpolated_with_demand.csv', index_col='timestamp', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S.%f')

#Smoothing the data
data['demand'] = data['demand'].rolling(window=10).mean()
data = data.dropna()

# Define the ARIMA model
model = ARIMA(data['demand'], order=(5,1,0))

# Fit the model
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

# Plot density of residuals
residuals.plot(kind='kde')
plt.show()

# Print residual errors
print(residuals.describe())

# Predict and compute error
X = data['demand'].values
size = int(len(X) * 0.67)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0), enforce_stationarity=False)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# Plot the data and predictions
plt.figure(figsize=(12,6))
plt.plot(data.index, data['demand'], color='blue', label='Actual')
plt.plot(data.index[size:], predictions, color='red', label='Predicted')
plt.axvline(x=data.index[size], color='green', linestyle='--', label='Start of Prediction')
plt.xlabel('Date')
plt.ylabel('Demand')
# plt.ylabel('Stock Value')
plt.title('ARIMA Model Predictions')
plt.legend()
plt.show()