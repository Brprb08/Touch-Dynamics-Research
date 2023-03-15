import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import math

def adjust_num_rows(data):
    # Find the minimum number of rows for each unique id in the 'class' column
    min_rows = data.groupby('class').size().min()

    

    # Adjust the number of rows for each unique id to match the minimum number of rows
    new_data = pd.DataFrame(columns=data.columns)
    i = 0
    for class_val in data['class'].unique():
        class_data = data[data['class'] == class_val]
        new_data = pd.concat([new_data, class_data.sample(min_rows)])
        
    return new_data
        
    


# Load the data into a DataFrame
#data = pd.read_csv('combined_fixed.csv')
data = pd.read_csv('combined_fixed.csv', header=0, dtype=float)
# Adjust the number of rows for each unique id  
data = adjust_num_rows(data)

# Extract the 'class' column from the data and convert it to a one-hot encoded array
labels = pd.get_dummies(data['class']).values

# Convert the labels to one-hot encoded format

# Remove the 'class' column from the data
features = data.drop(['class'], axis=1).values

# Split the data into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.4, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(features.shape[1],)),
    Dense(32, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(labels.shape[1], activation='softmax')
])


opt = SGD(lr=0.01)

# Compile the model with categorical crossentropy loss and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=100, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_features, test_labels)
print(f'Test loss: {loss:.3f}\nTest accuracy: {accuracy:.3f}')

# Make predictions on the test data
predictions = model.predict(test_features)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Convert one-hot encoded labels back to class labels
true_classes = np.argmax(test_labels, axis=1)

# Calculate F1 score and accuracy
from sklearn.metrics import f1_score, accuracy_score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
acc = accuracy_score(true_classes, predicted_classes)

print(f'F1 score: {f1:.3f}\nAccuracy: {acc:.3f}')