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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')
# print(f"List of seaborn datasets: \n{sns.get_dataset_names()}")
#
#
#
#
#
#
diamonds = sns.load_dataset('diamonds')
print(f"There are {diamonds.isna().sum().sum()} missing values")
diamonds
#
#
#
#
grouped = diamonds.groupby(['clarity', 'cut', 'color'])['price'].sum()

# Sort the grouped data within each 'clarity' group
grouped_sorted = grouped.reset_index().sort_values(['clarity', 'price'], ascending=[True, False])

# Unstack the sorted grouped data
grouped_sorted_unstacked = grouped_sorted.set_index(['clarity', 'cut', 'color']).unstack().fillna(0)

# Create a stacked bar plot with sorted bars
grouped_sorted_unstacked.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.legend(title='Color', labels=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
# Add x-label, y-label, and title
plt.xlabel('Clarity and Cut', fontname='serif', color='darkred',)
plt.ylabel('Total Price', fontname='serif', color='darkred',)
plt.title('Total Price by Clarity and Cut, Sorted within Each Clarity Group', fontname='serif', color='darkblue', fontsize=16)
plt.show()
#
#
#
#
#
#
df = diamonds.copy()
scaler = StandardScaler()
df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']] = scaler.fit_transform(df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']])
df
#
#
#
#
#
#
#
#
def fit(X, y):
  classes = np.unique(y)
  parameters = []
  for i, c in enumerate(classes):
    X_c = X[y == c]
    parameters.append({
      'prior': X_c.shape[0] / X.shape[0],
      'mean': X_c.mean(axis=0),
      'var': X_c.var(axis=0)
      })
  return classes, parameters

def predict(X, classes, parameters):
  N, D = X.shape
  K = len(classes)
  P = np.zeros((N, K))
  for k in range(K):
    P[:, k] = np.log(parameters[k]['prior'])
    P[:, k] += -0.5 * np.sum(np.log(2. * np.pi * parameters[k]['var']))
    P[:, k] += -0.5 * np.sum(((X - parameters[k]['mean']) ** 2) / (parameters[k]['var']), 1)
  return np.argmax(P, 1)
#
#
#
#
#
columns = ['cut', 'clarity', 'color']
columne_name = columns[0]
X = df[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']]
y = df[columne_name]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classes, parameters = fit(X_train, y_train)
predictions = predict(X_test, classes, parameters)
predictions_label = classes[predictions]
accuracy = accuracy_score(y_test, predictions_label)
print(f"Accuracy on {columne_name} is {accuracy:.2f}")
#
#
#
#
df = diamonds.copy()
scaler = StandardScaler()
df[['carat', 'depth', 'table', 'x', 'y', 'z']] = scaler.fit_transform(df[['carat', 'depth', 'table', 'x', 'y', 'z']])

X = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y = df[['cut', 'clarity', 'color']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classes_cut, parameters_cut = fit(X_train, y_train['cut'])
predictions_cut = predict(X_test, classes_cut, parameters_cut)

classes_clarity, parameters_clarity = fit(X_train, y_train['clarity'])
predictions_clarity = predict(X_test, classes_clarity, parameters_clarity)

classes_color, parameters_color = fit(X_train, y_train['color'])
predictions_color = predict(X_test, classes_color, parameters_color)
#
#
#
#
count_df = pd.DataFrame()
count_df['correct_cut'] = (y_test['cut'] == classes_cut[predictions_cut]).map({True: 'Correct', False: 'Incorrect'})
count_df['correct_clarity'] = (y_test['clarity'] == classes_clarity[predictions_clarity]).map({True: 'Correct', False: 'Incorrect'})
count_df['correct_color'] = (y_test['color'] == classes_color[predictions_color]).map({True: 'Correct', False: 'Incorrect'})

# Count the number of correct predictions for each feature
correct_cut = count_df['correct_cut'].value_counts()['Correct']
correct_clarity = count_df['correct_clarity'].value_counts()['Correct']
correct_color = count_df['correct_color'].value_counts()['Correct']

# Count the number of incorrect predictions for each feature
incorrect_cut = count_df['correct_cut'].value_counts()['Incorrect']
incorrect_clarity = count_df['correct_clarity'].value_counts()['Incorrect']
incorrect_color = count_df['correct_color'].value_counts()['Incorrect']

accuracy_cut = accuracy_score(y_test['cut'], classes_cut[predictions_cut])
accuracy_clarity = accuracy_score(y_test['clarity'], classes_clarity[predictions_clarity])
accuracy_color = accuracy_score(y_test['color'], classes_color[predictions_color])

# Create a DataFrame for the counts
data = {'Correct': [correct_cut, correct_clarity, correct_color],
        'Incorrect': [incorrect_cut, incorrect_clarity, incorrect_color]}
df_counts = pd.DataFrame(data, index=['cut', 'clarity', 'color'])

colors=plt.get_cmap('Paired', 2)

df_counts.plot(kind='barh', stacked=True, color=colors.colors)

accuracies = [accuracy_cut, accuracy_clarity, accuracy_color]
for i, (v, accuracy) in enumerate(zip(df_counts['Correct'], accuracies)):
  plt.text(v, i, f' {v} ({accuracy*100:.2f}%)', va='center')

plt.xlabel('Count', fontname='serif', color='darkred',)
plt.ylabel('Category', fontname='serif', color='darkred',)
plt.title('Prediction Correctness', fontname='serif', color='darkblue', fontsize=16)
plt.show()
#
#
#
#
#
from matplotlib_venn import venn3
import matplotlib.patches as mpatches

# Count the number of correct predictions for each pair of features
correct_cut_clarity = count_df[(count_df['correct_cut'] == 'Correct') & (count_df['correct_clarity'] == 'Correct')].shape[0]
correct_cut_color = count_df[(count_df['correct_cut'] == 'Correct') & (count_df['correct_color'] == 'Correct')].shape[0]
correct_clarity_color = count_df[(count_df['correct_clarity'] == 'Correct') & (count_df['correct_color'] == 'Correct')].shape[0]

# Count the number of correct predictions for all three features
correct_all = count_df[(count_df['correct_cut'] == 'Correct') & (count_df['correct_clarity'] == 'Correct') & (count_df['correct_color'] == 'Correct')].shape[0]

# Create the Venn diagram
plt.figure(figsize=(8,8))
venn = venn3(subsets=(correct_cut, correct_clarity, correct_color, correct_cut_clarity, correct_cut_color, correct_clarity_color, correct_all), set_labels=('Cut', 'Clarity', 'Color'))

for text in venn.set_labels:
  text.set_fontname('serif')
  text.set_color('darkred')

plt.title('Correct Predictions', fontname='serif', color='darkblue', fontsize=16)

# Create a custom legend
legend_elements = [mpatches.Patch(color=venn.get_patch_by_id('100').get_facecolor(), label='Cut'),
mpatches.Patch(color=venn.get_patch_by_id('010').get_facecolor(), label='Clarity'),
mpatches.Patch(color=venn.get_patch_by_id('001').get_facecolor(), label='Color')]

# Calculate the count of the union of correct predictions
correct_union = count_df[(count_df['correct_cut'] == 'Correct') | (count_df['correct_clarity'] == 'Correct') | (count_df['correct_color'] == 'Correct')].shape[0]

# Calculate the count of the complementary set
complement_count = len(y_test) - correct_union

# Add the count of the complementary set to the plot
plt.text(0.2, 0.5, f'Complementary set \nin Universe\n({complement_count})', horizontalalignment='center', verticalalignment='center')

plt.legend(handles=legend_elements, loc='best')

plt.show()
#
#
#
#
#
#
