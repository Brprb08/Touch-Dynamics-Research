import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

     # Define the neural network model
     model = Sequential()
     model.add(Dense(units=64, activation='sigmoid', input_dim=44))
     #model.add(Dropout(0.5))
     model.add(Dense(units=256, activation='sigmoid'))
     model.add(Dense(units=256, activation='sigmoid'))
     #model.add(Dropout(0.5))
     model.add(Dense(units=1, activation='sigmoid'))

     # Compile the model
     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

     # Count the number of positive and negative samples in the training data
     num_positive_samples = len(user_train_data)
     num_negative_samples = len(other_train_data)

     # Calculate the class weights based on the training data
     class_weights = {0: num_negative_samples / (num_positive_samples + num_negative_samples),
                 1: num_positive_samples / (num_positive_samples + num_negative_samples)}

     # Train the model on all data except the test data
     early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
     train_data = pd.concat([user_train_data, other_train_data], ignore_index=True)
     train_labels = train_data['class']
     #train_labels = train_labels.values
     train_labels = train_labels.map(lambda x: 0 if x == user_id else 1).values
     train_data.drop(['class'], axis=1, inplace=True)
     train_data = scaler.transform(train_data)
     model.fit(train_data, train_labels, epochs=50, batch_size=10, validation_split=0.2, class_weight=class_weights, callbacks=[early_stop])

     #callbacks=[early_stop]

     # Evaluate the model
     test_predictions = model.predict(test_data)
     test_predictions = (test_predictions > 0.5).astype(int)
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

