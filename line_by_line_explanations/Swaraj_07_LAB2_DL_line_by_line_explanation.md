# Swaraj_07_LAB2_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB2_DL.ipynb`

**Aim:** Simple and multiple linear regression for house price prediction, including preprocessing, model training, evaluation, and comparison.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from sklearn.model_selection import train_test_split` | Imports train_test_split for dividing data into training and testing sets. |
| 6 | `from sklearn.linear_model import LinearRegression` | Imports LinearRegression for simple and multiple regression models. |
| 7 | `from sklearn.metrics import mean_squared_error, r2_score` | Imports mean_squared_error for regression error calculation, r2_score for measuring regression fit quality. |
| 8 | `from sklearn.preprocessing import StandardScaler` | Imports StandardScaler for converting features to mean 0 and standard deviation 1. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `data = {` | Stores the dataset for later preprocessing and modeling. |
| 2 | `    'Area': [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500],` | Defines the values for the \`Area\` column in the DataFrame or output table. |
| 3 | `    'Bedrooms': [2, 3, 3, 4, 4, 5, 5, 6],` | Defines the values for the \`Bedrooms\` column in the DataFrame or output table. |
| 4 | `    'Age': [10, 8, 5, 12, 7, 3, 2, 1],` | Defines the values for the \`Age\` column in the DataFrame or output table. |
| 5 | `    'Price': [200000, 250000, 280000, 350000, 400000, 450000, 500000, 550000]` | Defines the values for the \`Price\` column in the DataFrame or output table. |
| 6 | `}` | Closes the multi-line statement or data structure started above. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `df = pd.DataFrame(data)` | Creates a Pandas DataFrame from the provided dictionary or table. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `print(df.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `print(df.isnull().sum())` | Prints the count of missing values in each column. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `df.fillna(df.mean(), inplace=True)` | Fills missing values with the column mean in place. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `scaler = StandardScaler()` | Creates a StandardScaler object for feature standardization. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `features = ['Area', 'Bedrooms', 'Age']` | Stores the feature column names that will be scaled. |
| 4 | `df[features] = scaler.fit_transform(df[features])` | Fits the scaler on the data and transforms the values into scaled form. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_simple = df[['Area']]   # Independent var` | Stores the input feature columns used by the model. |
| 2 | `y = df['Price']           # Dependent var` | Stores the target/output values that the model must predict. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_train, X_test, y_train, y_test = train_test_split(` | Splits features and target into training and testing datasets. |
| 2 | `    X_simple, y, test_size=0.2, random_state=42` | Passes the data and split settings into train_test_split. |
| 3 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model_simple = LinearRegression()` | Creates a linear regression model. |
| 2 | `model_simple.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `y_pred_simple = model_simple.predict(X_test)` | Generates predictions using the trained model. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `mse_simple = mean_squared_error(y_test, y_pred_simple)` | Calculates mean squared error between actual and predicted values. |
| 2 | `print("MSE:", mse_simple)` | Prints the mean squared error value. |

## Code Cell 14

| Line | Code | Explanation |
|---:|---|---|
| 1 | `rmse_simple = np.sqrt(mse_simple)` | Takes the square root, converting MSE into RMSE. |
| 2 | `print("RMSE:", rmse_simple)` | Prints the mean squared error value. |

## Code Cell 15

| Line | Code | Explanation |
|---:|---|---|
| 1 | `r2_simple = r2_score(y_test, y_pred_simple)` | Calculates the R2 score to measure regression fit. |
| 2 | `print("R2 Score:", r2_simple)` | Prints the R2 score value. |

## Code Cell 16

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.scatter(X_simple, y, color='blue')` | Creates a scatter plot to show the relationship between two variables. |
| 2 | `plt.plot(X_simple, model_simple.predict(X_simple), color='red')` | Plots values as a line graph for visualization. |
| 3 | `plt.xlabel("Area")` | Labels the x-axis of the graph. |
| 4 | `plt.ylabel("Price")` | Labels the y-axis of the graph. |
| 5 | `plt.title("Simple Linear Regression")` | Adds a title to the current graph. |
| 6 | `plt.show()` | Displays the completed plot. |

## Code Cell 18

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_multi = df[['Area', 'Bedrooms', 'Age']]` | Stores the input feature columns used by the model. |
| 2 | `y = df['Price']` | Stores the target/output values that the model must predict. |

## Code Cell 19

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_train, X_test, y_train, y_test = train_test_split(` | Splits features and target into training and testing datasets. |
| 2 | `    X_multi, y, test_size=0.2, random_state=42` | Passes the data and split settings into train_test_split. |
| 3 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 20

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model_multi = LinearRegression()` | Creates a linear regression model. |
| 2 | `model_multi.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |

## Code Cell 21

| Line | Code | Explanation |
|---:|---|---|
| 1 | `y_pred_multi = model_multi.predict(X_test)` | Generates predictions using the trained model. |

## Code Cell 22

| Line | Code | Explanation |
|---:|---|---|
| 1 | `mse_multi = mean_squared_error(y_test, y_pred_multi)` | Calculates mean squared error between actual and predicted values. |
| 2 | `print("MSE:", mse_multi)` | Prints the mean squared error value. |

## Code Cell 23

| Line | Code | Explanation |
|---:|---|---|
| 1 | `rmse_multi = np.sqrt(mse_multi)` | Takes the square root, converting MSE into RMSE. |
| 2 | `print("RMSE:", rmse_multi)` | Prints the mean squared error value. |

## Code Cell 24

| Line | Code | Explanation |
|---:|---|---|
| 1 | `r2_multi = r2_score(y_test, y_pred_multi)` | Calculates the R2 score to measure regression fit. |
| 2 | `print("R2 Score:", r2_multi)` | Prints the R2 score value. |

## Code Cell 25

| Line | Code | Explanation |
|---:|---|---|
| 1 | `comparison = pd.DataFrame({` | Creates a Pandas DataFrame from the provided dictionary or table. |
| 2 | `    'Actual': y_test,` | Defines the values for the \`Actual\` column in the DataFrame or output table. |
| 3 | `    'Predicted': y_pred_multi` | Defines the values for the \`Predicted\` column in the DataFrame or output table. |
| 4 | `})` | Closes the multi-line statement or data structure started above. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `print(comparison)` | Prints the given message or variable value for inspection. |

