# Swaraj_07_LAB4_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB4_DL.ipynb`

**Aim:** Gaussian Naive Bayes classification on the Social Network Ads dataset with label encoding and model evaluation.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 2 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `from sklearn.model_selection import train_test_split` | Imports train_test_split for dividing data into training and testing sets. |
| 5 | `from sklearn.naive_bayes import GaussianNB` | Imports GaussianNB for Naive Bayes classification with Gaussian feature distribution. |
| 6 | `from sklearn.preprocessing import LabelEncoder` | Imports LabelEncoder for converting text categories into numbers. |
| 7 | `from sklearn.metrics import accuracy_score, confusion_matrix, classification_report` | Imports accuracy_score for classification accuracy, confusion_matrix for summarizing correct and wrong class predictions, classification_report for precision, recall, F1-score, and support. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `data = pd.read_csv("Social_Network_Ads.csv")` | Reads the CSV dataset into a Pandas DataFrame. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("First 10 rows of dataset:\n")` | Prints the given message or variable value for inspection. |
| 4 | `print(data.head(10))` | Prints the first few rows so the dataset can be checked. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `print("\nDataset Shape:", data.shape)` | Prints the shape/size to verify data dimensions. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `print("\nDataset Information:\n")` | Prints the given message or variable value for inspection. |
| 9 | `print(data.info())` | Prints the given message or variable value for inspection. |
| 10 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `data = data.drop("User ID", axis=1)` | Removes User ID because it is only an identifier and not useful for prediction. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("\nDataset after removing User ID:\n")` | Prints the given message or variable value for inspection. |
| 4 | `print(data.head())` | Prints the first few rows so the dataset can be checked. |
| 5 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `encoder = LabelEncoder()` | Creates a LabelEncoder for converting categories into numeric labels. |
| 2 | `data["Gender"] = encoder.fit_transform(data["Gender"])` | Learns category mapping and converts Gender values into numeric codes. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `print("\nEncoded Dataset:\n")` | Prints the given message or variable value for inspection. |
| 5 | `print(data.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X = data.drop("Purchased", axis=1)` | Separates the input features by dropping the target column Purchased. |
| 2 | `y = data["Purchased"]` | Stores the target/output values that the model must predict. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `print("\nFeatures (X):\n", X.head())` | Prints the first few rows so the dataset can be checked. |
| 5 | `print("\nTarget (y):\n", y.head())` | Prints the first few rows so the dataset can be checked. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_train, X_test, y_train, y_test = train_test_split(` | Splits features and target into training and testing datasets. |
| 2 | `    X, y,` | Passes the feature matrix and target vector to `train_test_split`. |
| 3 | `    test_size=0.25,` | Sets the fraction of data reserved for testing. |
| 4 | `    random_state=0` | Fixes randomness so results are reproducible. |
| 5 | `)` | Closes the multi-line statement or data structure started above. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `print("\nTraining Data Size:", X_train.shape)` | Prints the shape/size to verify data dimensions. |
| 8 | `print("Testing Data Size:", X_test.shape)` | Prints the shape/size to verify data dimensions. |
| 9 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = GaussianNB()` | Creates a Gaussian Naive Bayes classifier. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `model.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `print("\nModel Training Completed")` | Prints the given message or variable value for inspection. |
| 6 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `y_pred = model.predict(X_test)` | Generates predictions using the trained model. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("\nPredicted Output:\n", y_pred)` | Prints the given message or variable value for inspection. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `print("\nActual Output:\n", y_test.values)` | Prints the given message or variable value for inspection. |
| 6 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `accuracy = accuracy_score(y_test, y_pred)` | Calculates classification accuracy by comparing actual and predicted labels. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("\nAccuracy of Naive Bayes Classifier:", accuracy * 100, "%")` | Prints the calculated accuracy value. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `cm = confusion_matrix(y_test, y_pred)` | Creates a confusion matrix from true and predicted labels. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("\nConfusion Matrix:\n")` | Prints the given message or variable value for inspection. |
| 4 | `print(cm)` | Prints the confusion matrix showing correct and incorrect classifications. |
| 5 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `print("\nClassification Report:\n")` | Prints the given message or variable value for inspection. |
| 2 | `print(classification_report(y_test, y_pred))` | Prints the classification report with precision, recall, F1-score, and support. |

