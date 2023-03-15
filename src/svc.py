from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('combined_fixed.csv')

# Find the id with the least number of rows
min_rows = np.inf
for i in range(1, 16):
    rows = df.loc[df['class'] == i].shape[0]
    if rows < min_rows:
        min_rows = rows

# Split the data so that each class has the same number of rows
dfs = []
for i in range(1, 16):
    temp_df = df.loc[df['class'] == i].sample(n=min_rows, random_state=42)
    dfs.append(temp_df)

new_df = pd.concat(dfs)


# Split the data into features and target
X = new_df.drop('class', axis=1)
y = new_df['class']

# Train the SVM classifier
clf = svm.SVC()
clf.fit(X, y)

# Make predictions on the training set
y_pred = clf.predict(X)

# Evaluate the performance of the classifier
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

print(f'Accuracy: {acc}')
print(f'F1 score: {f1}')