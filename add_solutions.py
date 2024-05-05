import pandas as pd
import numpy as np
import os

# Function to add a column indicating the numerical stock price movement
def add_price_movement(df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate the difference in 'Adj Close' price from the previous day
    df['Price_Diff'] = df['Adj Close'].diff()

    return df

# Base directory for cleaned data
cleaned_base_dir = 'cleaned_data'

# New base directory for updated data with solution
updated_base_dir = 'data_with_solution'

# Subdirectories to iterate over
sub_dirs = ['stocks', 'etfs']

# Iterate over each subdirectory and process the CSV files
for sub_dir in sub_dirs:
    folder_path = os.path.join(cleaned_base_dir, sub_dir)
    updated_folder_path = os.path.join(updated_base_dir, sub_dir)
    
    # Create the subdirectory in the updated data directory if it doesn't exist
    os.makedirs(updated_folder_path, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            updated_file_path = os.path.join(updated_folder_path, filename)
            df = pd.read_csv(file_path)

            # Add the numerical price movement column
            df_updated = add_price_movement(df)

            # Save the updated data to the new file system
            df_updated.to_csv(updated_file_path, index=False)

            print(f'Updated data for {filename} with numerical price movement information and saved to {updated_file_path}')
