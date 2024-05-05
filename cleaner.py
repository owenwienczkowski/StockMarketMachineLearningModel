import pandas as pd
import numpy as np
import os
from scipy import stats

# Function to clean and preprocess the data
def clean_data(df):
    # Create a copy of the DataFrame to manipulate
    df = df.copy()

    # Handling missing values using forward fill
    df.ffill(inplace=True)

    # Identifying and handling outliers using Z-score
    z_scores = stats.zscore(df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]

    # Data consistency checks
    # Ensuring date format consistency (example format: YYYY-MM-DD)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Feature engineering
    # Example: Creating a feature for daily return percentage
    df['Daily_Return'] = df['Adj Close'].pct_change() * 100

    # Normalization/Scaling
    # Example: Scaling the 'Volume' feature
    df['Volume'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()

    # Handling non-stationarity by differencing
    df['Adj Close Diff'] = df['Adj Close'].diff()

    return df

# Base directory containing the 'stocks' and 'etfs' folders
base_dir = 'Data'

# New base directory for cleaned data
cleaned_base_dir = 'cleaned_data'

# Subdirectories to iterate over
sub_dirs = ['stocks', 'etfs']

# Iterate over each subdirectory and process the CSV files
for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    cleaned_folder_path = os.path.join(cleaned_base_dir, sub_dir)
    
    # Create the subdirectory in the cleaned data directory if it doesn't exist
    os.makedirs(cleaned_folder_path, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            cleaned_file_path = os.path.join(cleaned_folder_path, filename)
            df = pd.read_csv(file_path)

            # Clean and preprocess the data
            df_cleaned = clean_data(df)

            # Save the cleaned data to the new file system
            df_cleaned.to_csv(cleaned_file_path, index=False)

            print(f'Processed and cleaned data for {filename} and saved to {cleaned_file_path}')
