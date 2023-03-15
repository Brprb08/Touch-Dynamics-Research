import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np
from xgboost import plot_tree


# Load the data into a DataFrame
df = pd.read_csv('combined_fixed.csv')

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.33)


# Convert the data to DMatrix format, which is the input format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the hyperparameters for the model
params = {
    'objective': 'multi:softmax',
    'num_class': 16,
    'max_depth': 140,
    'learning_rate': 1,
    'silent': 1,
    'n_estimators': 200
}

#evals = [(dtrain, 'Train'), (dtest, 'Test')]

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=25, evals=[(dtrain, 'train')])

# Make predictions on the test set
predictions = bst.predict(dtest)

print(predictions)

# Evaluate the model
accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

results = bst.get_score(importance_type='weight')
#epochs = len(results['validation_0']['merror'])
#x_axis = range(0, epochs)
fig = plt.subplots()

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix: \n", cm)

# F1 Score
f1_score = classification_report(y_test, predictions, output_dict=True)['weighted avg']['f1-score']
print("F1 Score: %.2f" % f1_score)


plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
xgb.plot_tree(bst)
plt.show()

# Plot the feature importances
xgb.plot_importance(bst)
plt.show()



cm = confusion_matrix(y_test, predictions)
#tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]


false_negative_rate = fn / (fn + tp)

# Calculate false positive rate
false_positive_rate = fp / (tn + fp)

# Calculate equal error rate
eer = min(false_negative_rate, false_positive_rate)

print("False Negative Rate: ", false_negative_rate)
print("False Positive Rate: ", false_positive_rate)
print("Equal Error Rate: ", eer)
