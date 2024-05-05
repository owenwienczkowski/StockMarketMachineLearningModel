import pandas as pd
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Base path to the 'Data' folder
base_folder_path = 'small_data_with_solution'

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

# print(combined_df.head())  # This will print the first five rows of the combined dataset


# Assuming 'data' is your preprocessed DataFrame with the 'Adj Close' prices
# Calculate the 'Price_Diff' column as the target variable
combined_df['Price_Diff'] = combined_df['Adj Close'].diff()

# Handle the NaN value in the first row after differencing
combined_df.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_df['Price_Diff'].values.reshape(-1, 1))

# Prepare the sequences for LSTM with 'Price_Diff' as the target
def create_sequences(data, sequence_length=10):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
        ys.append(data[i + sequence_length])
    return np.array(xs), np.array(ys)

sequence_length = 10  # Number of days to look back for prediction
X, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
# For regression, we typically don't use 'accuracy' as a metric, so it's omitted here
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Evaluate the model using the test set
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = predicted_stock_price

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Print the evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Print the loss and validation loss
print(f"Loss: {history.history['loss'][-1]}")
print(f"Validation Loss: {history.history['val_loss'][-1]}")
