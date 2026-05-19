# Swaraj_07_LAB9_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB9_DL.ipynb`

**Aim:** LSTM-based stock price prediction using both a CSV dataset and real stock data downloaded with yfinance.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from sklearn.preprocessing import MinMaxScaler` | Imports MinMaxScaler for scaling values into a fixed range. |
| 6 | `from tensorflow.keras.models import Sequential` | Imports Sequential for creating layer-by-layer neural networks. |
| 7 | `from tensorflow.keras.layers import Dense, LSTM` | Imports Dense for fully connected neural network layers, LSTM for sequence and time-series learning. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `data = pd.read_csv('stock_prices.csv')` | Reads the CSV dataset into a Pandas DataFrame. |
| 2 | `data = data[['Close']]` | Stores the dataset for later preprocessing and modeling. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `scaler = MinMaxScaler(feature_range=(0,1))` | Creates a MinMaxScaler object to scale values into a fixed range. |
| 2 | `scaled_data = scaler.fit_transform(data)` | Fits the scaler on the data and transforms the values into scaled form. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X = []` | Stores the input feature columns used by the model. |
| 2 | `y = []` | Stores the target/output values that the model must predict. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `for i in range(60, len(scaled_data)):` | Starts a loop that repeats the indented block for each item in the range or sequence. |
| 5 | `    X.append(scaled_data[i-60:i, 0])` | Adds the selected value or sequence to the list. |
| 6 | `    y.append(scaled_data[i, 0])` | Adds the selected value or sequence to the list. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `X, y = np.array(X), np.array(y)` | Converts lists into NumPy arrays for numerical processing. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `# Reshape for RNN` | Comment explaining this step: Reshape for RNN. |
| 11 | `X = np.reshape(X, (X.shape[0], X.shape[1], 1))` | Reshapes data into the 3D format required by an LSTM/RNN. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `split = int(0.8 * len(X))` | Calculates the index used to divide data into training and testing portions. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `X_train, X_test = X[:split], X[split:]` | Stores the computed value in \`X_train, X_test\` for use in later steps. |
| 4 | `y_train, y_test = y[:split], y[split:]` | Stores the computed value in \`y_train, y_test\` for use in later steps. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1)))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 4 | `model.add(LSTM(units=50))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `model.add(Dense(1))  # output layer` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `model.compile(optimizer='adam', loss='mean_squared_error')` | Configures the model with optimizer, loss function, and metrics. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(X_train, y_train, epochs=10, batch_size=32)` | Trains/fits the model or estimator using the provided data. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `predictions = model.predict(X_test)` | Generates predictions using the trained model. |
| 2 | `predictions = scaler.inverse_transform(predictions)` | Converts scaled values back to the original real-world scale. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `actual = scaler.inverse_transform(y_test.reshape(-1,1))` | Converts scaled values back to the original real-world scale. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `plt.plot(actual, label='Actual Price')` | Plots values as a line graph for visualization. |
| 4 | `plt.plot(predictions, label='Predicted Price')` | Plots values as a line graph for visualization. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `plt.title("Stock Price Prediction")` | Adds a title to the current graph. |
| 7 | `plt.xlabel("Time")` | Labels the x-axis of the graph. |
| 8 | `plt.ylabel("Price")` | Labels the y-axis of the graph. |
| 9 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `plt.show()` | Displays the completed plot. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `pip install yfinance` | Installs the required Python package(s): yfinance. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import yfinance as yf` | Imports yfinance to download real stock market data directly from Yahoo Finance. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `# Example: Apple stock` | Comment explaining this step: Example: Apple stock. |
| 5 | `df = yf.download('AAPL', start='2018-01-01', end='2024-01-01')` | Downloads historical Apple stock data for the given date range. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `# Save dataset` | Comment explaining this step: Save dataset. |
| 8 | `df.to_csv("real_stock_data.csv")` | Saves the downloaded stock data as a CSV file. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `print(df.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 14

| Line | Code | Explanation |
|---:|---|---|
| 1 | `from sklearn.preprocessing import MinMaxScaler` | Imports MinMaxScaler for scaling values into a fixed range. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `data = df[['Open','High','Low','Close','Volume']]` | Stores the dataset for later preprocessing and modeling. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `scaler = MinMaxScaler()` | Creates a MinMaxScaler object to scale values into a fixed range. |
| 6 | `scaled_data = scaler.fit_transform(data)` | Fits the scaler on the data and transforms the values into scaled form. |

## Code Cell 15

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X, y = [], []` | Initializes or stores the input sequences and target values. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `for i in range(60, len(scaled_data)):` | Starts a loop that repeats the indented block for each item in the range or sequence. |
| 4 | `    X.append(scaled_data[i-60:i])` | Adds the selected value or sequence to the list. |
| 5 | `    y.append(scaled_data[i, 3])  # Close price` | Adds the selected value or sequence to the list. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `X, y = np.array(X), np.array(y)` | Converts lists into NumPy arrays for numerical processing. |

## Code Cell 16

| Line | Code | Explanation |
|---:|---|---|
| 1 | `from tensorflow.keras.layers import Dropout` | Imports Dropout for reducing overfitting. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `model = Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 6 | `model.add(Dropout(0.2))` | Adds dropout regularization to reduce overfitting. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `model.add(LSTM(64))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 9 | `model.add(Dropout(0.2))` | Adds dropout regularization to reduce overfitting. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `model.add(Dense(25))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 12 | `model.add(Dense(1))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 13 | `` | Blank line used to separate code logically and improve readability. |
| 14 | `model.compile(optimizer='adam', loss='mean_squared_error')` | Configures the model with optimizer, loss function, and metrics. |

## Code Cell 17

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(X, y, epochs=20, batch_size=32)` | Trains/fits the model or estimator using the provided data. |

## Code Cell 18

| Line | Code | Explanation |
|---:|---|---|
| 1 | `pred = model.predict(X)` | Generates predictions using the trained model. |
| 2 | `plt.plot(y, label='Actual')` | Plots values as a line graph for visualization. |
| 3 | `plt.plot(pred, label='Predicted')` | Plots values as a line graph for visualization. |
| 4 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 5 | `plt.title("Real vs Predicted")` | Adds a title to the current graph. |
| 6 | `plt.show()` | Displays the completed plot. |

## Code Cell 19

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['loss'])` | Plots values as a line graph for visualization. |
| 2 | `plt.title("Training Loss")` | Adds a title to the current graph. |
| 3 | `plt.show()` | Displays the completed plot. |

## Code Cell 20

| Line | Code | Explanation |
|---:|---|---|
| 1 | `from sklearn.metrics import mean_squared_error` | Imports mean_squared_error for regression error calculation. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `rmse = np.sqrt(mean_squared_error(y, pred))` | Takes the square root, converting MSE into RMSE. |
| 4 | `print("RMSE:", rmse)` | Prints the mean squared error value. |

