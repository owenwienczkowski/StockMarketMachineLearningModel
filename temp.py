# import pandas as pd
# import os

# # Path to the folder containing your CSV files
# folder_path = 'stocks'

# # List to hold all the individual DataFrames
# dataframes_list = []

# # Loop through all the CSV files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path)
#         dataframes_list.append(df)

# # Concatenate all the DataFrames into a single DataFrame
# combined_df = pd.concat(dataframes_list, ignore_index=True)

# print(combined_df.head())  # This will print the first five rows of the combined dataset


import pandas as pd

# Assuming 'AA.csv' is one of the files in the "stocks" folder
file_path = 'data/etfs/AAAU.csv'
df = pd.read_csv(file_path)

print(df.head())  # This will display the first five rows of the DataFrame
