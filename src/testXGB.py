import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('pubg_train_merge.csv', header=None)

# Replace the column name in the code to the actual column name in the dataframe
#column_names = ['mean_x_speed','mean_y_speed','mean_speed','mean_x_acc','mean_y_acc','mean_acc','mean_jerk','mean_tan','mean_ang','mean_touch_major','mean_touch_minor','std_x_speed','std_y_speed','std_speed','std_x_acc','std_y_acc','std_acc','std_ang','std_tan','std_jerk','std_touch_major','std_touch_minor','min_x_speed','min_y_speed','min_speed','min_x_acc','min_y_acc','min_acc','min_ang','min_tan','min_jerk','min_touch_major','min_touch_minor','max_x_speed','max_y_speed','max_speed','max_x_acc','max_y_acc','max_acc','max_ang','max_tan,max_jerk','max_touch_major','max_touch_minor','class'
#]
#df.columns = column_names

df = df.iloc[1:]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.33)

# Convert the data to DMatrix format, which is the input format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the hyperparameters for the model
params = {
    'objective': 'multi:softmax',
    'num_class': 15, # change the number of classes to the correct number of unique user IDs
    'max_depth': 15,
    'learning_rate': 0.7,
    'silent': 1,
    'n_estimators': 100
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=10, evals=[(dtrain, 'train')])

# Make predictions on the test set
predictions = bst.predict(dtest)

# Evaluate the model
accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

# Plot the feature importances
xgb.plot_importance(bst)
plt.show()