import os
# environment variable recommended for formatting and consistency.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Base path to the labeled data folder. Switch to "small_data_with_solution" for a quicker runtime.
base_folder_path = 'small_data_with_solution'
# base_folder_path = 'medium_data_with_solution'


# Subfolders within the data folder
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
            
            # Add an identifier column to know which file the data came from
            df['Source'] = filename
            
            dataframes_list.append(df)

# Concatenate all the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes_list, ignore_index=True)

# print(combined_df.head())  # This will print the first five rows of the combined dataset


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the input layer
input_layer = Input(shape=(sequence_length, 1))

# Build the LSTM layers
lstm_out = Bidirectional(LSTM(units=50, return_sequences=True))(input_layer)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = Bidirectional(LSTM(units=50, return_sequences=True))(lstm_out)

# Apply the Attention mechanism
query_encoding = Dense(128)(lstm_out)
value_encoding = Dense(128)(lstm_out)
attention_out = Attention()([query_encoding, value_encoding])

# Final Dense layer for prediction
output_layer = Dense(1)(attention_out)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2, callbacks=[callback])

# Predictions
predicted_stock_price = model.predict(X_test)
# Reshape from 3D to 2D
predicted_stock_price_2d = predicted_stock_price.reshape(-1, predicted_stock_price.shape[2])

predicted_stock_price_inversed = scaler.inverse_transform(predicted_stock_price_2d)

predicted_stock_price = predicted_stock_price_inversed.reshape(predicted_stock_price.shape[0], predicted_stock_price.shape[1], -1)
# Evaluate the model using the test set
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = predicted_stock_price

# Print the model's guess and the correct solution for each prediction
for i in range(len(y_pred)):
    # Extract the scalar value from the array for formatting
    model_guess = y_pred[i][0].item()  # Convert numpy float to Python scalar
    correct_solution = y_true[i][0].item()  # Convert numpy float to Python scalar
    print(f"Model's guess: {model_guess:.4f}, Correct solution: {correct_solution:.4f}")

# Reshape y_pred to 2D
y_pred_2d = y_pred[:, -1, 0]  # Taking the last timestep and first feature

mae = mean_absolute_error(y_true, y_pred_2d)
mse = mean_squared_error(y_true, y_pred_2d)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred_2d)

# Print the evaluation metrics
print(f"MAE: {mae}") # work towards < 1%
print(f"MSE: {mse}") # work towards 0
print(f"RMSE: {rmse}") # work towards < 1%
print(f"R-squared: {r2}") # work towards 1


# Print the loss and validation loss
print(f"Loss: {history.history['loss'][-1]}") # keep low
print(f"Validation Loss: {history.history['val_loss'][-1]}") # work towards training loss

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot the actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_true, label='Actual Price')
plt.plot(y_pred_2d, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.ylabel('Price')
plt.xlabel('Time Index')
plt.legend(loc='upper left')
plt.show()