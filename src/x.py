import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb

# Load the CSV file
data = pd.read_csv('diep_correct.csv')

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

for i in range(1,16):

     # Extract the data for the user you want to test on
     user_id = i # Replace 1 with the user ID you want to test on
     user_data = data[data['class'] == user_id].reset_index(drop=True)

     # Extract the data for random users
     other_data = data[data['class'] != user_id].reset_index(drop=True)

     # Split the user data into training and testing sets
     user_train_data, user_test_data = train_test_split(user_data, test_size=0.3)

     # Split the other data into training and testing sets
     other_train_data, other_test_data = train_test_split(other_data, test_size=0.3)

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

     num_positive_samples = len(user_train_data)
     num_negative_samples = len(other_train_data)

     # Define the XGBoost model
     model = xgb.XGBClassifier(
         max_depth=30,
         learning_rate=0.6,
         n_estimators=100,
         objective='binary:logistic',
         booster='gbtree',
         n_jobs=-1,
         scale_pos_weight=num_negative_samples / num_positive_samples
     )

     # Train the model on all data except the test data
     train_data = pd.concat([user_train_data, other_train_data], ignore_index=True)
     train_labels = train_data['class']
     train_labels = train_labels.map(lambda x: 0 if x == user_id else 1).values
     train_data.drop(['class'], axis=1, inplace=True)
     train_data = scaler.transform(train_data)
     model.fit(train_data, train_labels)

     # Evaluate the model
     test_predictions = model.predict(test_data)
     accuracy = accuracy_score(test_labels, test_predictions)
     f1 = f1_score(test_labels, test_predictions)
     tn, fp, fn, tp = confusion_matrix(test_labels, test_predictions).ravel()
     fpr = fp / (fp + tn)
     fnr = fn / (fn + tp)

     # Print the evaluation results
     print("Accuracy: %", accuracy * 100)
     print("F1 Score:", f1)
     print("False Positive Rate: %", fpr * 100)
     print("False Negative Rate: %", fnr * 100)

     # Print the confusion matrix
     print("Confusion Matrix:")
     print(confusion_matrix(test_labels, test_predictions))