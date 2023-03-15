import os
import pandas as pd
import re

dir_path1 = r'c:\Users\pelto\OneDrive\Desktop\Research Code\snake_train'
dir_path2 = r'c:\Users\pelto\OneDrive\Desktop\Research Code\snake_test'
save_dir = r'c:\Users\pelto\OneDrive\Desktop\Research Code\snake_appended'

# create dictionary to store dataframes by number
df_dict = {}

# loop through files in first directory
for file in os.listdir(dir_path1):
    # check if file is a CSV file
    if not file.endswith('.csv'):
        continue
    
    # extract number from filename using regular expression
    num = re.search(r'Sub(\d+)Snake', file).group(1)
    
    # read in csv file as dataframe
    df = pd.read_csv(os.path.join(dir_path1, file))
    
    # add dataframe to dictionary using number as key
    key = f"Sub{num}"
    if key in df_dict:
        df_dict[key]['train'] = pd.concat([df_dict[key]['train'], df])
    else:
        df_dict[key] = {'train': df, 'test': None}

# loop through files in second directory
for file in os.listdir(dir_path2):
    # check if file is a CSV file
    if not file.endswith('.csv'):
        continue
    
    # extract number from filename using regular expression
    num = re.search(r'Sub(\d+)Snake', file).group(1)
    
    # read in csv file as dataframe
    df = pd.read_csv(os.path.join(dir_path2, file))
    
    # add dataframe to dictionary using number as key
    key = f"Sub{num}"
    if key in df_dict:
        df_dict[key]['test'] = pd.concat([df_dict[key]['test'], df])
    else:
        df_dict[key] = {'train': None, 'test': df}

# save dataframes to files
for key, dfs in df_dict.items():
    if dfs['train'] is not None and dfs['test'] is not None:
        combined = pd.concat([dfs['train'], dfs['test']])
        combined.to_csv(os.path.join(save_dir, f"{key}_combined.csv"), index=False)