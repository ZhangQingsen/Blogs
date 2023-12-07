# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| output: false

import warnings
warnings.filterwarnings("ignore")
#
#
#
#
#
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
#
#
#
#
#
#
#| echo: false
#| output: false
def my_mae(y_pred, y_test):
  assert len(y_pred) == len(y_test)
  m = len(y_pred)
  res = np.abs(y_test-y_pred)
  return res.mean()

def my_mse(y_pred, y_test):
  assert len(y_pred) == len(y_test)
  m = len(y_pred)
  res = (y_test-y_pred) ** 2
  return res.mean()

def my_rmse(y_pred, y_test):
  assert len(y_pred) == len(y_test)
  m = len(y_pred)
  res = (y_test-y_pred) ** 2
  return (res.mean()) ** .5

def my_mad(y_pred, y_test):
  assert len(y_pred) == len(y_test)
  m = len(y_pred)
  res = np.abs(y_pred-np.median(y_test))
  return np.median(res)

def linearPredict(x, ws):
  ones = np.ones([x.shape[0], 1])
  x_1 = np.hstack((x, ones))
  y_pred = np.dot(x_1,ws)
  return y_pred

def linear_update_weigths(x, y, w, lr):
  yMat = np.mat(y).T
  y_pred = np.dot(x,w)
  gradient = np.array(lr * 2 * np.dot(x.T, (y_pred - y).T)).flatten()
  w -= gradient
  y_pred = np.dot(x,w)
  return w, y_pred

def to_power(x, power):
  x = x.reshape(x.shape[0], -1)
  x_tmp = x[:]
  for i in range(2, power+1):
      x = np.append(x, x_tmp ** i, axis=1)
  return x

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def logLoss(y, h):
  return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def normalize_and_binarize(y):
  # y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
  # y_binarized = np.where(y_standardized >= 0.5, 1, 0)
  y_standardized = (y - np.mean(y)) / np.std(y)
  y_binarized = np.where(y_standardized >= 0, 1, 0)
  return y_binarized

def logistic_update_weigths(x, y, w, lr):
  h = sigmoid(np.dot(x, w))
  loss = logLoss(y, h)
  gradient = np.dot(x.T, (h - y)) / y.shape[0]
  w -= lr *gradient
  h = sigmoid(np.dot(x, w))
  return w, h

def fit_linear(x, y, lr=0.001, num_iter=2000, func=my_rmse):
  perform_df = pd.DataFrame(columns=['loss'])
  ones = np.ones([x.shape[0]]).reshape(-1,1)
  x_1 = np.hstack((x, ones))
  weight = np.random.rand(x_1.shape[1])  # initialize weight
  for epoch in range(num_iter):
    weight, y_pred = linear_update_weigths(x_1, y, weight, lr)
    # weight, y_pred = linear_update_weigths_1(x_1, y, weight, lr, func)
    loss = func(y, y_pred)
    perform_df.loc[epoch] = [loss]
  return weight, y_pred, perform_df

def fit_logistic(x, y, lr=0.001, num_iter=2000):
  perform_df = pd.DataFrame(columns=['loss'])
  ones = np.ones([x.shape[0], 1])
  x_1 = np.hstack((x, ones))
  weight = np.random.rand(x_1.shape[1])  # initialize weight
  for epoch in range(num_iter):
    weight, h = logistic_update_weigths(x_1, y, weight, lr)
    loss = logLoss(y, h)
    perform_df.loc[epoch] = [loss]
  return weight, h, perform_df
```
#
#
#
#
#
#
#
#
# stock = 'AAPL'
stock = 'GOOGL'
period = '1d'
start = '2000-1-1'
end = '2022-12-31'

tickerData = yf.Ticker(stock)

tickerDf = tickerData.history(period=period, start=start, end=end)

tickerDf.index = pd.to_datetime(tickerDf.index)

tickerDf
#
#
#
#
close_arr = tickerDf['Close']

fig = plt.figure()
plt.plot(close_arr.index, close_arr, label=stock)
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel("Price($)")
plt.title(f"{stock} price VS. Date")
plt.legend()
plt.plot()
#
#
#
#
#
return_arr = tickerDf['Close'].pct_change().dropna()
return_arr = (1+return_arr).cumprod()

fig = plt.figure()
plt.plot(return_arr.index, return_arr, label=stock)
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel("Cumulative\nReturn")
plt.title(f"{stock} return VS. Date")
plt.legend()
plt.plot()
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
df = pd.DataFrame()
df['return'] = tickerDf['Close'].pct_change()

# make sure not using today's data to predict today's result
df['day_momentum'] = (1 + df['return']).shift(1) - 1
df['week_momentum'] = (1 + df['return']).rolling(window=5).apply(np.prod, raw=True).shift(1) - 1
df['month_momentum'] = (1 + df['return']).rolling(window=21).apply(np.prod, raw=True).shift(1) - 1
df['quarter_momentum'] = (1 + df['return']).rolling(window=63).apply(np.prod, raw=True).shift(1) - 1
df['year_momentum'] = (1 + df['return']).rolling(window=252).apply(np.prod, raw=True).shift(1) - 1
df['52_weeks_high_low_ratio'] = (tickerDf['Close'] - tickerDf['Low'].rolling(window=252).min()) / (tickerDf['High'].rolling(window=252).max() - tickerDf['Low'].rolling(window=252).min())
df
#
#
#
#
#
#
#
# Smoothing with a moving average
df['day_momentum_smooth'] = df['day_momentum'].rolling(window=5).mean()
df['week_momentum_smooth'] = df['week_momentum'].rolling(window=5).mean()
df['month_momentum_smooth'] = df['month_momentum'].rolling(window=5).mean()
df['quarter_momentum_smooth'] = df['quarter_momentum'].rolling(window=5).mean()
df['year_momentum_smooth'] = df['year_momentum'].rolling(window=5).mean()

# Risk adjustment
df['day_momentum_risk_adj'] = df['day_momentum'] / df['return'].rolling(window=252).std()
df['week_momentum_risk_adj'] = df['week_momentum'] / df['return'].rolling(window=252).std()
df['month_momentum_risk_adj'] = df['month_momentum'] / df['return'].rolling(window=252).std()
df['quarter_momentum_risk_adj'] = df['quarter_momentum'] / df['return'].rolling(window=252).std()
df['year_momentum_risk_adj'] = df['year_momentum'] / df['return'].rolling(window=252).std()
#
#
#
#
scaler = StandardScaler()
# scaler = MinMaxScaler()
df_scaled = df.dropna()
return_arr = df_scaled['return']
df_scaled = df_scaled[df_scaled.columns[1:]]
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df.columns[1:])
df_scaled.set_index(return_arr.index, inplace=True)
df_scaled
#
#
#
#
#
for e in df_scaled.columns:
  factor = df_scaled[e]
  corr_coef = np.corrcoef(factor, return_arr)[0, 1]
  if np.abs(corr_coef) > 0.1:
    print(f"The correlation coefficient between {e} and target is {corr_coef:.2f}")

#
#
#
#
#
#
X = df_scaled
y_linear = return_arr
y_logistic = normalize_and_binarize(return_arr)

X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X, y_linear, test_size=0.3, shuffle=False)
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X, y_logistic, test_size=0.3, shuffle=False)
#
#
#
#
#
#
# Fit the linear model
learning_rate = 0.00001
iterations = 1000

weights_linear, y_pred_linear, perform_df_linear = fit_linear(X_train_linear.values, y_train_linear.values, lr=learning_rate, num_iter=iterations)

loss = my_rmse(y_train_linear.values, y_pred_linear)
y_train_bin = normalize_and_binarize(y_train_linear.values)
y_pred_bin = normalize_and_binarize(y_pred_linear)
accu = accuracy_score(y_train_bin, y_pred_bin)
print(f"Final loss:{loss:.2f}, accuracy: {accu:.2f}")

x_axis = X_train_linear.index
y_signal = np.where(y_train_bin == y_pred_bin, 1, -1)

fig = plt.figure()
plt.bar(x_axis, y_signal)
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel("Prediction\nOutcome")
plt.title("Performance in Linear regression Training")
# plt.legend()
plt.show()
# Evaluate the models
# ... (depends on how you want to evaluate your models)
#
#
#
#
#
# Fit the logistic model
learning_rate = 0.01
iterations = 1000

weights_logistic, h_pred_logistic, perform_df_logistic = fit_logistic(X_train_logistic.values, y_train_logistic, lr=learning_rate, num_iter=iterations)

y_pred_logistic = normalize_and_binarize(h_pred_logistic)

loss = logLoss(y_train_logistic, h_pred_logistic)
accu = accuracy_score(y_train_logistic, y_pred_logistic)
print(f"Final loss:{loss:.2f}, accuracy: {accu:.2f}")

x_axis = X_train_logistic.index
y_signal = np.where(y_train_logistic == y_pred_logistic, 1, -1)

fig = plt.figure()
plt.bar(x_axis, y_signal)
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel("Prediction\nOutcome")
plt.title("Performance in Logistic regression Training")
# plt.legend()
plt.show()
#
#
#
#
#
#
y_pred_linear = linearPredict(X_test_linear, weights_linear)
y_pred_logistic = linearPredict(X_test_logistic, weights_logistic)

test_return_arr = y_test_linear

# calculate the return
y_pred_linear_return = np.where(y_pred_linear > 0, test_return_arr, -test_return_arr)
y_pred_logistic_return = np.where(y_pred_logistic > 0, test_return_arr, -test_return_arr)
x_axis = test_return_arr.index

cum_test_return = (1+test_return_arr).cumprod()
cum_linear_return = (1+y_pred_linear_return).cumprod()
cum_logistic_return = (1+y_pred_logistic_return).cumprod()


fig = plt.figure()
plt.plot(x_axis, cum_test_return, label='test_return')
plt.plot(x_axis, cum_linear_return, label='test_return')
plt.plot(x_axis, cum_logistic_return, label='test_return')
plt.xlabel("Date")
plt.xticks(rotation=30)
plt.ylabel("Cumulative\nReturn")
plt.title("Strategy comparison on testset")
plt.legend()
plt.show()
#
#
#
