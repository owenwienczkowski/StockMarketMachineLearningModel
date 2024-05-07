# StockMarketMachineLearningModel

In the realm of finance, the ability to accurately predict the future movement of stock prices can result in significant economic gains. However, the stock market is influenced by a myriad of factors, including economic indicators, company performance, and investor sentiment, making it highly volatile and unpredictable. Traditional analysis methods have struggled to consistently forecast market trends and movements.

This project aims to harness the power of machine learning and data science to develop a predictive model that can analyze historical stock market data and identify patterns that may indicate future price movements. By leveraging advanced algorithms and computational techniques, the goal is to create a system that can provide more accurate predictions than traditional methods, thus offering valuable insights for investors and traders. More specifically, the model should be able to predict the direction of the stock for the next day.

The specific objectives of the project are to:

1. Collect and preprocess a comprehensive dataset of historical stock prices, along with relevant financial indicators.
2. Explore and analyze the data to understand the key factors that influence stock market trends.
3. Develop and train machine learning models using the processed data to predict future stock prices.
4. Evaluate the performance of the models using appropriate metrics and compare them with baseline predictors.
5. Implement a user-friendly interface that allows users to input specific stocks and receive predictions along with confidence intervals.
6. The successful completion of this project could revolutionize the approach to stock market investments and open new avenues for financial analysis and decision-making.

The dataset used to train this model can be accessed here:
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset


## The code is designed to predict future stock prices using historical data. It does this by:

Gathering Data: It combines stock and ETF data from different files into one large dataset.
Preparing Data: It calculates the day-to-day price changes and normalizes these values to be between 0 and 1, which helps the model learn more effectively.
Creating Sequences: It organizes the data into sequences, each sequence being a set of consecutive days’ data, to use for predictions.
Building a Model: It constructs a neural network model that can learn from these sequences. This model is made up of layers designed to recognize patterns in the data.
Training the Model: The model is trained with a portion of the data, learning to predict the next day’s price change.
Testing the Model: The model’s predictions are compared against actual data it hasn’t seen before to evaluate its performance.
The results after running the code are metrics that tell us how well the model is performing:

MAE (Mean Absolute Error): On average, the model’s predictions are off by this amount.
MSE (Mean Squared Error): A higher value indicates that the model’s predictions are widely varied from the actual prices.
RMSE (Root Mean Squared Error): Similar to MSE, but more representative of the actual error because it’s in the same units as the original data.
R-squared: This tells us the percentage of the variance in the target variable that the model is able to predict.


### Cleaning process:
For this dataset, etfs/PRN.csv was removed for formatting reasons
To resolve empty spaces, forward feed was used to provide the most recent value since stocks were documented chronologically.
Outliers are removed (using z-score).
Adjusted date formats for data consistency.
Engineered Adj. Close to be the feature of focus.
Normalized the volume feature.
Differenced for non-stationarity.
Created 'cleaned_data' folder to store results of cleaned stocks and cleaned etfs.
Finally, using the Adj. Close, the price difference in stock is calculated and used to validate the model. The dataset with this column added is "cleaned_data_with_solution". 

This allows for the machine learning model to properly train and be adequately validated.

### To run the model:
Run the "Engine.py" program. The parameters can all be adjusted to check the adaptability of the model. For simple testing, it is recommended to run the model on the largely reduced "cleaned_data_with_solution" data set as this will significantly reduce runtime (at the expense of results). If you have powerful enough hardware, feel free to run the entire "data_with_solution" data set.