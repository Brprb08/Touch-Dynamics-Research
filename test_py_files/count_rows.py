import pandas as pd

# read in dataset as dataframe
df = pd.read_csv(r'c:\Users\pelto\OneDrive\Desktop\Research Code\mc_correct.csv')

# create empty dictionary to hold counts for each user
user_counts = {}

# loop through each unique user in 'class' column
for user in df['class'].unique():
    # count number of rows for user
    user_count = df[df['class'] == user].shape[0]
    # add count to dictionary
    user_counts[user] = user_count

# print counts for each user
for user, count in user_counts.items():
    print(f"User {user} has {count} rows.")