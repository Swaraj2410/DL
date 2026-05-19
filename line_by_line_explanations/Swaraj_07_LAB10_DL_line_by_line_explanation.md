# Swaraj_07_LAB10_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB10_DL.ipynb`

**Aim:** LSTM-based Delhi temperature time-series forecasting using historical climate data and future prediction.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `pip install numpy pandas matplotlib scikit-learn tensorflow` | Installs the required Python package(s): numpy pandas matplotlib scikit-learn tensorflow. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from sklearn.preprocessing import MinMaxScaler` | Imports MinMaxScaler for scaling values into a fixed range. |
| 6 | `from tensorflow.keras.models import Sequential` | Imports Sequential for creating layer-by-layer neural networks. |
| 7 | `from tensorflow.keras.layers import LSTM, Dense` | Imports LSTM for sequence and time-series learning, Dense for fully connected neural network layers. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `df = pd.read_csv("DailyDelhiClimateTrain.csv")` | Reads the CSV dataset into a Pandas DataFrame. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print(df.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# Select only required columns` | Comment explaining this step: Select only required columns. |
| 2 | `df = df[['date', 'meantemp']]` | Keeps only the date and mean temperature columns needed for forecasting. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `# Rename columns` | Comment explaining this step: Rename columns. |
| 5 | `df.columns = ['Date', 'Temperature']` | Renames the selected columns to clearer names. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `# Convert date column` | Comment explaining this step: Convert date column. |
| 8 | `df['Date'] = pd.to_datetime(df['Date'])` | Converts date strings into Pandas datetime values. |
| 9 | `df.set_index('Date', inplace=True)` | Sets the Date column as the index so the data acts like a time series. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `print(df.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure(figsize=(10,5))` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `plt.plot(df['Temperature'], label='Temperature')` | Plots values as a line graph for visualization. |
| 3 | `plt.title("Delhi Temperature Over Time")` | Adds a title to the current graph. |
| 4 | `plt.xlabel("Date")` | Labels the x-axis of the graph. |
| 5 | `plt.ylabel("Temperature")` | Labels the y-axis of the graph. |
| 6 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 7 | `plt.show()` | Displays the completed plot. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `scaler = MinMaxScaler()` | Creates a MinMaxScaler object to scale values into a fixed range. |
| 2 | `scaled_data = scaler.fit_transform(df[['Temperature']])` | Fits the scaler on the data and transforms the values into scaled form. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `def create_dataset(data, time_step=30):` | Defines a reusable function for creating time-series input/output samples. |
| 2 | `    X, y = [], []` | Initializes or stores the input sequences and target values. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `    for i in range(len(data) - time_step):` | Starts a loop that repeats the indented block for each item in the range or sequence. |
| 5 | `        X.append(data[i:i+time_step])` | Adds the selected value or sequence to the list. |
| 6 | `        y.append(data[i+time_step])` | Adds the selected value or sequence to the list. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `    return np.array(X), np.array(y)` | Converts lists into NumPy arrays for numerical processing. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `time_step = 30` | Stores an important hyperparameter used later in the notebook. |
| 11 | `X, y = create_dataset(scaled_data, time_step)` | Initializes or stores the input sequences and target values. |
| 12 | `` | Blank line used to separate code logically and improve readability. |
| 13 | `print(X.shape, y.shape)` | Prints the shape/size to verify data dimensions. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `train_size = int(len(X) * 0.8)` | Calculates the index used to divide data into training and testing portions. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `X_train, X_test = X[:train_size], X[train_size:]` | Stores the computed value in \`X_train, X_test\` for use in later steps. |
| 4 | `y_train, y_test = y[:train_size], y[train_size:]` | Stores the computed value in \`y_train, y_test\` for use in later steps. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 4 | `model.add(LSTM(50))` | Adds an LSTM layer to learn sequence/time-series patterns. |
| 5 | `model.add(Dense(1))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `model.compile(optimizer='adam', loss='mean_squared_error')` | Configures the model with optimizer, loss function, and metrics. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `model.summary()` | Prints the neural network architecture and parameter counts. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(` | Trains/fits the model or estimator using the provided data. |
| 2 | `    X_train, y_train,` | Passes the training input and target arrays to `model.fit`. |
| 3 | `    epochs=20,` | Sets the number of complete training passes through the dataset. |
| 4 | `    batch_size=32,` | Sets how many samples are processed before each weight update. |
| 5 | `    validation_data=(X_test, y_test)` | Provides validation data to monitor performance during training. |
| 6 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['loss'], label='Training Loss')` | Plots values as a line graph for visualization. |
| 2 | `plt.plot(history.history['val_loss'], label='Validation Loss')` | Plots values as a line graph for visualization. |
| 3 | `plt.title("Model Loss")` | Adds a title to the current graph. |
| 4 | `plt.xlabel("Epochs")` | Labels the x-axis of the graph. |
| 5 | `plt.ylabel("Loss")` | Labels the y-axis of the graph. |
| 6 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 7 | `plt.show()` | Displays the completed plot. |

## Code Cell 14

| Line | Code | Explanation |
|---:|---|---|
| 1 | `train_predict = model.predict(X_train)` | Generates predictions using the trained model. |
| 2 | `test_predict = model.predict(X_test)` | Generates predictions using the trained model. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `# Convert back to original values` | Comment explaining this step: Convert back to original values. |
| 5 | `train_predict = scaler.inverse_transform(train_predict)` | Converts scaled values back to the original real-world scale. |
| 6 | `test_predict = scaler.inverse_transform(test_predict)` | Converts scaled values back to the original real-world scale. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `y_train_actual = scaler.inverse_transform(y_train)` | Converts scaled values back to the original real-world scale. |
| 9 | `y_test_actual = scaler.inverse_transform(y_test)` | Converts scaled values back to the original real-world scale. |

## Code Cell 15

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure(figsize=(12,6))` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# Plot actual data` | Comment explaining this step: Plot actual data. |
| 4 | `plt.plot(df.index, df['Temperature'], label='Actual', linewidth=2)` | Plots values as a line graph for visualization. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `# Train prediction index` | Comment explaining this step: Train prediction index. |
| 7 | `train_dates = df.index[time_step: time_step + len(train_predict)]` | Stores the computed value in \`train_dates\` for use in later steps. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `# Test prediction index` | Comment explaining this step: Test prediction index. |
| 10 | `test_dates = df.index[time_step + len(train_predict): time_step + len(train_predict) + len(test_predict)]` | Stores the computed value in \`test_dates\` for use in later steps. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `# Plot predictions` | Comment explaining this step: Plot predictions. |
| 13 | `plt.plot(train_dates, train_predict, label='Train Prediction')` | Plots values as a line graph for visualization. |
| 14 | `plt.plot(test_dates, test_predict, label='Test Prediction')` | Plots values as a line graph for visualization. |
| 15 | `` | Blank line used to separate code logically and improve readability. |
| 16 | `plt.title("LSTM Temperature Prediction (Delhi)")` | Adds a title to the current graph. |
| 17 | `plt.xlabel("Date")` | Labels the x-axis of the graph. |
| 18 | `plt.ylabel("Temperature")` | Labels the y-axis of the graph. |
| 19 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 20 | `plt.grid()` | Adds grid lines to make the graph easier to read. |
| 21 | `` | Blank line used to separate code logically and improve readability. |
| 22 | `plt.show()` | Displays the completed plot. |

## Code Cell 16

| Line | Code | Explanation |
|---:|---|---|
| 1 | `future_steps = 7` | Stores an important hyperparameter used later in the notebook. |
| 2 | `last_data = scaled_data[-time_step:]` | Stores the computed value in \`last_data\` for use in later steps. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `future_predictions = []` | Stores predicted output values generated by the model. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `for _ in range(future_steps):` | Starts a loop that repeats the indented block for each item in the range or sequence. |
| 7 | `    pred = model.predict(last_data.reshape(1, time_step, 1))` | Reshapes the array into the dimensions expected by the model. |
| 8 | `    future_predictions.append(pred[0][0])` | Adds the selected value or sequence to the list. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `    last_data = np.append(last_data[1:], pred, axis=0)` | Updates the rolling time-series window with the newest prediction. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `# Convert back to original scale` | Comment explaining this step: Convert back to original scale. |
| 13 | `future_predictions = scaler.inverse_transform(` | Converts scaled values back to the original real-world scale. |
| 14 | `    np.array(future_predictions).reshape(-1,1)` | Converts lists into NumPy arrays for numerical processing. |
| 15 | `)` | Closes the multi-line statement or data structure started above. |
| 16 | `` | Blank line used to separate code logically and improve readability. |
| 17 | `print("Next 7 Days Temperature Prediction:")` | Prints the given message or variable value for inspection. |
| 18 | `print(future_predictions)` | Prints the given message or variable value for inspection. |

## Code Cell 17

| Line | Code | Explanation |
|---:|---|---|
| 1 | `future_dates = pd.date_range(` | Creates a sequence of dates for future forecast plotting. |
| 2 | `    start=df.index[-1],` | Sets the starting date for the generated date range. |
| 3 | `    periods=future_steps+1,` | Sets how many dates to generate. |
| 4 | `    freq='D'` | Sets the date frequency to daily. |
| 5 | `)[1:]` | Closes the date_range call and removes the first date so only future dates remain. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `plt.figure(figsize=(10,5))` | Creates a new plotting figure, optionally with a specified size. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `plt.plot(df.index, df['Temperature'], label='Historical')` | Plots values as a line graph for visualization. |
| 10 | `plt.plot(future_dates, future_predictions, label='Future Forecast', marker='o')` | Plots values as a line graph for visualization. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `plt.title("Future Temperature Prediction (Delhi)")` | Adds a title to the current graph. |
| 13 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 14 | `plt.show()` | Displays the completed plot. |

