import os
import pandas as pd
import numpy as np

def fix_csv(file_path):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)

    # Replace -inf values with -1000
    df.replace(['-inf', 'inf', np.inf, -np.inf], [-1000, -1000, -1000, -1000], inplace=True)

    df.fillna(-1000, inplace=True)

    df = df.astype(int)

    # Save the fixed dataframe to the same file
    df.to_csv(file_path, index=False, sep=',')

directory = r'c:\Users\pelto\OneDrive\Desktop\Research Code\diep_test_data'

# Get the list of all files in the current working directory
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Loop through every file in the list
for file in files:
    file_path = os.path.join(directory, file)
    fix_csv(file_path)


# Concatenate all the fixed CSV files into one file
combined_df = pd.concat([pd.read_csv(os.path.join(directory, f))for f in files], ignore_index=True)

# Save the combined dataframe to a new file
#combined_df.to_csv("combined_fixed_pubg_test.csv", index=False, sep=',')