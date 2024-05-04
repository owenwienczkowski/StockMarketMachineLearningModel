import pandas as pd
import os

# Base path to the 'Data' folder
base_folder_path = 'Small_Data'

# Subfolders within the 'Data' folder
subfolders = ['stocks', 'etfs']

# List to hold all the individual DataFrames
dataframes_list = []

# Loop through each subfolder
for subfolder in subfolders:
    # Full path to the subfolder
    folder_path = os.path.join(base_folder_path, subfolder)
    
    # Loop through all the CSV files in the subfolder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # You can add an identifier column to know which file the data came from
            df['Source'] = filename
            
            dataframes_list.append(df)

# Concatenate all the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes_list, ignore_index=True)

print(combined_df.head())  # This will print the first five rows of the combined dataset
