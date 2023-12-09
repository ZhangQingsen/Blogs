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
#
#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')
# print(f"List of seaborn datasets: \n{sns.get_dataset_names()}")
#
#
#
#
#
url="https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/mnist_test.csv"
minst_df = pd.read_csv(url)
minst_df.shape
#
#
#
#
rows = minst_df.iloc[:100].copy(deep=True)
rows.sort_values(by="label",ascending=True, inplace=True)
plt.figure(figsize=(12,12))
plt.suptitle("MNIST Data Preview", fontname='serif', color='darkblue', fontsize=16)
for i in range(100):
  row = rows.iloc[i]
  pic = row[1:].values.reshape(28,28)
  plt.subplot(10,10,i+1)
  plt.imshow(pic)
plt.show()
#
#
#
#
X = minst_df.drop(columns=['label']).values
y = minst_df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
#
#
#
#
x1 = pd.DataFrame(X)
rows = x1.iloc[:10].copy(deep=True)
# rows.sort_values(by="label",ascending=True, inplace=True)
plt.figure(figsize=(12,2))
plt.suptitle("MNIST Data after Nromalization", fontname='serif', color='darkblue', fontsize=16)
for i in range(10):
  row = rows.iloc[i]
  pic = row.values.reshape(28,28)
  plt.subplot(1,10,i+1)
  plt.imshow(pic)
plt.show()
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
def sigmoid(x):
  x = np.clip(x, -500, 500)
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def compute_loss(y, y_hat):
  return np.mean((y - y_hat)**2)

def forward_propagation(X, weights, biases):
  z = np.dot(X, weights) + biases
  return sigmoid(z)

def back_propagation(X, y, y_hat, weights, learning_rate):
  error = y - y_hat
  d_weights = np.dot(X.T, error * sigmoid_derivative(y_hat))
  d_weights = np.mean(d_weights, axis=1, keepdims=True)
  weights += learning_rate * d_weights
  return weights

def fit(X, y, epochs, learning_rate):
  weights = np.random.rand(X.shape[1], 1)
  biases = np.zeros((1,))

  for epoch in range(epochs):
    y_hat = forward_propagation(X, weights, biases)
    loss = compute_loss(y, y_hat)
    weights = back_propagation(X, y, y_hat, weights, learning_rate)

    if epoch % 100 == 0:
      print(f"Epoch {epoch}, Loss: {loss:.2f}")

  return weights, biases

def predict(X, weights, biases):
    return forward_propagation(X, weights, biases)

#
#
#
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
weights, biases = fit(X_train, y_train, epochs=10, learning_rate=0.1)
y_train_pred = predict(X_train, weights, biases)
y_test_pred = predict(X_test, weights, biases)

train_accuracy = accuracy_score(y_train, np.round(y_train_pred.flatten()).astype(np.int8))
test_accuracy = accuracy_score(y_test, np.round(y_test_pred.flatten()).astype(np.int8))

print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
#
#
#
#
x_train1 = pd.DataFrame(X_train)
rows = x_train1.iloc[:10].copy(deep=True)
plt.figure(figsize=(12,2))
plt.suptitle("Predicted Labels in train dataset", fontname='serif', color='darkblue', fontsize=16)
for i in range(10):
  row = rows.iloc[i]
  pic = row.values.reshape(28,28)
  plt.subplot(1,10,i+1)
  plt.xlabel(f"true:{y_train[i]}\npred:{np.round(y_train_pred.flatten()[i]).astype(np.int8)}")
  plt.imshow(pic)
plt.show()
#
#
#
#
x_test1 = pd.DataFrame(X_test)
rows = x_train1.iloc[:10].copy(deep=True)
plt.figure(figsize=(12,2))
plt.suptitle("Predicted Labels in test dataset", fontname='serif', color='darkblue', fontsize=16)
for i in range(10):
  row = rows.iloc[i]
  pic = row.values.reshape(28,28)
  plt.subplot(1,10,i+1)
  plt.xlabel(f"true:{y_train[i]}\npred:{np.round(y_train_pred.flatten()[i]).astype(np.int8)}")
  plt.imshow(pic)
plt.show()
#
#
#
#
#
#
