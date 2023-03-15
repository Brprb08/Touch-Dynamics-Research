import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# load data
data = pd.read_csv('combined_fixed.csv')

# split data into training and testing
train_data = pd.DataFrame()
test_data = pd.DataFrame()
for user in data['class'].unique():
    user_data = data[data['class'] == user]
    user_train, user_test = train_test_split(user_data, test_size=0.2)
    train_data = train_data.append(user_train)
    test_data = test_data.append(user_test)

# create feature and label arrays
X_train = train_data.drop(['class', 'class'], axis=1).values
y_train = pd.get_dummies(train_data['class']).values
X_test = test_data.drop(['class', 'class'], axis=1).values
y_test = pd.get_dummies(test_data['class']).values

# define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# make predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# calculate and print accuracy and f1 score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
print('Accuracy:', accuracy)
print('F1 Score:', f1)

# get feature importance
weights = model.get_weights()[0]
feature_importance = pd.DataFrame(weights.T, index=train_data.columns[:-2], columns=train_data['class'].unique())
print('Feature Importance:\n', feature_importance)