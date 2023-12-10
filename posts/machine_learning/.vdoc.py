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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
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

# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))
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
#
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def cross_entropy(y, y_hat):
  m = y.shape[0]
  y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
  log_likelihood = -np.log(y_hat[range(m), y])
  loss = np.sum(log_likelihood) / m
  return loss

# update weight
def back_propagation(X, y, y_hat, weights, learning_rate):
  error = y_hat
  error[range(y.shape[0]), y] -= 1
  d_weights = np.dot(X.T, error)
  weights -= learning_rate * d_weights
  return weights

# predict y_hat
def forward_propagation(X, weights, biases):
  z = np.dot(X, weights) + biases
  return softmax(z)

def fit(X, y, epochs, learning_rate):
  weights = np.random.rand(X.shape[1], 10) * 0.01
  biases = np.zeros((10,))

  for epoch in range(epochs):
    y_hat = forward_propagation(X, weights, biases)
    loss = cross_entropy(y, y_hat)
    weights = back_propagation(X, y, y_hat, weights, learning_rate)

    if epoch % 100 == 0:
      print(f"Epoch {epoch}, Loss: {loss:.2f}")
      print(f"Weights: {weights}")
      print(f"Predictions: {y_hat}")

  return weights, biases

def predict(X, weights, biases):
  return np.argmax(forward_propagation(X, weights, biases), axis=1)

#
#
#
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weights, biases = fit(X_train, np.argmax(y_train, axis=1), epochs=300, learning_rate=0.001)

y_train_pred = predict(X_train, weights, biases)
y_test_pred = predict(X_test, weights, biases)

train_accuracy = accuracy_score(np.argmax(y_train, axis=1), y_train_pred)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)

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
  plt.xlabel(f"True: {np.argmax(y_train[i])}\nPred: {y_train_pred[i]}")
  plt.imshow(pic)
plt.show()
#
#
#
#
x_test1 = pd.DataFrame(X_test)
rows = x_test1.iloc[:10].copy(deep=True)
plt.figure(figsize=(12,2))
plt.suptitle("Predicted Labels in test dataset", fontname='serif', color='darkblue', fontsize=16)
for i in range(10):
  row = rows.iloc[i]
  pic = row.values.reshape(28,28)
  plt.subplot(1,10,i+1)
  plt.xlabel(f"True: {np.argmax(y_train[i])}\nPred: {y_train_pred[i]}")
  plt.imshow(pic)
plt.show()
#
#
#
#
#
#
