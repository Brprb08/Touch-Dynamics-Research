import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

# Load the CSV file
data = pd.read_csv('diep_data.csv')

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Extract the data for the user you want to test on
user_id = 1 # Replace 1 with the user ID you want to test on
user_data = data[data['class'] == user_id].reset_index(drop=True)

# Extract the data for random users
other_data = data[data['class'] != user_id].reset_index(drop=True)

# Split the user data into training and testing sets
user_train_data, user_test_data = train_test_split(user_data, test_size=0.2)

# Split the other data into training and testing sets
other_train_data, other_test_data = train_test_split(other_data, test_size=0.2)

# Concatenate the user test data with a random sample of the other test data
test_data = pd.concat([user_test_data, other_test_data.sample(frac=0.4)], ignore_index=True)

# Extract the labels for the test data
test_labels = test_data['class']
test_labels = test_labels.map(lambda x: 0 if x == user_id else 1).values

# Drop the labels from the features data
test_data.drop(['class'], axis=1, inplace=True)

# Normalize the features data
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)

# Count the number of positive and negative samples in the training data
num_positive_samples = len(user_train_data)
num_negative_samples = len(other_train_data)

# Calculate the class weights based on the training data
class_weights = {0: num_negative_samples / (num_positive_samples + num_negative_samples),
                 1: num_positive_samples / (num_positive_samples + num_negative_samples)}

# Train the model on all data except the test data
train_data = pd.concat([user_train_data, other_train_data], ignore_index=True)
train_labels = train_data['class']
#train_labels = train_labels.values
train_labels = train_labels.map(lambda x: 0 if x == user_id else 1).values
train_data.drop(['class'], axis=1, inplace=True)
train_data = scaler.transform(train_data)

# Define the SVC model
model = SVC(class_weight=class_weights)

# Fit the model
model.fit(train_data, train_labels)

# Predict the test labels
test_predictions = model.predict(test_data)

# Evaluate the model
accuracy = accuracy_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)
tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

# Print the evaluation results
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("False Positive Rate:", fpr)
print("False Negative Rate:", fnr)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_predictions))