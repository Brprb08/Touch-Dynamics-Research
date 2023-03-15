import pandas as pd

def find_first_non_float(df):
    for col in df.columns:
        for val in df[col]:
            if isinstance(val, float):
                continue
            if isinstance(val, str) and val.lower() == 'nan':
                continue
            print(f"First non-float value in column {col}: {val}")
            return

# create a sample dataframe
df = pd.read_csv(r'c:\Users\pelto\OneDrive\Desktop\Research Code\pubg_raw\pubg1_touch.csv')

# call the function to find the first non-float value
find_first_non_float(df)