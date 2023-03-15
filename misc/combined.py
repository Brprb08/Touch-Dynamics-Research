import os
import pandas as pd

# set directory path
dir_path = r'c:\Users\pelto\OneDrive\Desktop\Research Code\mc_appended'

# create empty list to hold dataframes
df_list = []

# loop through files in directory
for file in os.listdir(dir_path):
    if file.endswith('.csv'):
        # read in csv file as dataframe
        df = pd.read_csv(os.path.join(dir_path, file))
        # append dataframe to list
        df_list.append(df)

# concatenate all dataframes into one
df_concat = pd.concat(df_list)

# find dataframe with smallest number of rows
min_rows = min(df_concat.shape[0] for df_concat in df_list)

# loop through dataframes and trim rows if necessary
for i, df in enumerate(df_list):
    if df.shape[0] > min_rows:
        df_list[i] = df.head(min_rows)

# concatenate all dataframes into one again
df_concat = pd.concat(df_list)

# save concatenated dataframe as csv in current working directory
df_concat.to_csv('mc_correct.csv', index=False)