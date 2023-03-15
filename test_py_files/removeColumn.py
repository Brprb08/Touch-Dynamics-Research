import os
import pandas as pd

dir_path = r'c:\Users\pelto\OneDrive\Desktop\Research Code\diep_raw'
column_to_remove = 'WIDTH_MINOR'

for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(dir_path, filename)
        df = pd.read_csv(filepath)
        df.drop(column_to_remove, axis=1, inplace=True)
        df.to_csv(filepath, index=False)