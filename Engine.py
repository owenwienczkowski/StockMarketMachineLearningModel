import os
# environment variable recommended for formatting and consistency.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.regularizers import l1_l2, l2
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Base path to the labeled data folder. Switch to "small_data_with_solution" for a quicker runtime.
base_folder_path = 'small_data_with_solution'
# base_folder_path = 'data_with_solution'


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
scaler = StandardScaler()
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

# Add a 1D Convolutional layer for feature extraction
conv_out = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
conv_out = MaxPooling1D(pool_size=2)(conv_out)
conv_out = Flatten()(conv_out)

# Build the LSTM layers
lstm_out = Bidirectional(LSTM(units=125, return_sequences=True))(input_layer)
lstm_out = Dropout(0.5)(lstm_out)  # Increase dropout rate to 0.5
lstm_out = Bidirectional(LSTM(units=125, return_sequences=True))(lstm_out)

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
    if epoch < 2:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Define the K-Fold Cross Validator
kfold = KFold(n_splits=2, shuffle=True)

# Initialize variables to store cross-validation results
cv_mae, cv_mse, cv_rmse, cv_r2 = [], [], [], []

# Loop through the indices the split() method returns
for train_index, test_index in kfold.split(X):
    # Generate training and testing sets for each fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    # Add Gaussian noise to input data
    noise_factor = 0.1
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    # Fit the model
    history = model.fit(X_train_noisy, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Predictions
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price_2d = predicted_stock_price.reshape(-1, predicted_stock_price.shape[2])
    predicted_stock_price_inversed = scaler.inverse_transform(predicted_stock_price_2d)
    predicted_stock_price = predicted_stock_price_inversed.reshape(predicted_stock_price.shape[0], predicted_stock_price.shape[1], -1)

    # Evaluate the model using the test set
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = predicted_stock_price[:, -1, 0]  # Taking the last timestep and first feature

    # Calculate evaluation metrics for the current fold
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Append metrics to the lists
    cv_mae.append(mae)
    cv_mse.append(mse)
    cv_rmse.append(rmse)
    cv_r2.append(r2)

# Calculate the average of the cross-validation metrics
avg_mae = np.mean(cv_mae)
avg_mse = np.mean(cv_mse)
avg_rmse = np.mean(cv_rmse)
avg_r2 = np.mean(cv_r2)

# Print the average cross-validation metrics
print(f"Average MAE: {avg_mae}")
print(f"Average MSE: {avg_mse}")
print(f"Average RMSE: {avg_rmse}")
print(f"Average R-squared: {avg_r2}")


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
plt.plot(y_pred, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.ylabel('Price')
plt.xlabel('Time Index')
plt.legend(loc='upper left')
plt.show()