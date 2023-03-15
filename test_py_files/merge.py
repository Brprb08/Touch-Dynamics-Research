import csv
import os

def merge_csv_files(directory, output_file):
    # Get list of all csv files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Create new csv file for the merged data
    with open(output_file, 'w', newline='') as merged_file:
        writer = csv.writer(merged_file)
        
        # Write header row
        header_written = False
        
        # Loop through each csv file in the directory
        for file in csv_files:
            with open(os.path.join(directory, file), 'r') as csv_file:
                reader = csv.reader(csv_file)
                
                # Write header row only once
                if not header_written:
                    for row in reader:
                        writer.writerow(row)
                        header_written = True
                        break
                # Write the rest of the rows
                for row in reader:
                    writer.writerow(row)
    

    return "Merged csv file saved as: {}".format(output_file)

# Example usage
merged_file = merge_csv_files(r'c:\Users\pelto\OneDrive\Desktop\Research Code\pubg_test', 'pubg_test_merge.csv')
print(merged_file)