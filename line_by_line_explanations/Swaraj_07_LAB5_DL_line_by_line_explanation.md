# Swaraj_07_LAB5_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB5_DL.ipynb`

**Aim:** Decision Tree, pruning analysis, Random Forest, and AdaBoost classification on the Social Network Ads dataset.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# Import libraries` | Comment explaining this step: Import libraries. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from sklearn.model_selection import train_test_split` | Imports train_test_split for dividing data into training and testing sets. |
| 6 | `from sklearn.preprocessing import LabelEncoder` | Imports LabelEncoder for converting text categories into numbers. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `from sklearn.tree import DecisionTreeClassifier` | Imports DecisionTreeClassifier for tree-based classification. |
| 9 | `from sklearn.metrics import accuracy_score, classification_report` | Imports accuracy_score for classification accuracy, classification_report for precision, recall, F1-score, and support. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `from sklearn.ensemble import RandomForestClassifier` | Imports RandomForestClassifier for an ensemble of decision trees. |
| 12 | `from sklearn.ensemble import AdaBoostClassifier` | Imports AdaBoostClassifier for boosting weak learners. |
| 13 | `` | Blank line used to separate code logically and improve readability. |
| 14 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 15 | `` | Blank line used to separate code logically and improve readability. |
| 16 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 1 Load Dataset` | Comment explaining this step: 1 Load Dataset. |
| 2 | `data = pd.read_csv("Social_Network_Ads.csv")` | Reads the CSV dataset into a Pandas DataFrame. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `print("Dataset Preview")` | Prints the given message or variable value for inspection. |
| 5 | `print(data.head())` | Prints the first few rows so the dataset can be checked. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `print("\nDataset Shape:", data.shape)` | Prints the shape/size to verify data dimensions. |
| 8 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 2 Data Preprocessing` | Comment explaining this step: 2 Data Preprocessing. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `data = data.drop("User ID", axis=1)` | Removes User ID because it is only an identifier and not useful for prediction. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `encoder = LabelEncoder()` | Creates a LabelEncoder for converting categories into numeric labels. |
| 6 | `data["Gender"] = encoder.fit_transform(data["Gender"])` | Learns category mapping and converts Gender values into numeric codes. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `print("\nProcessed Dataset")` | Prints the given message or variable value for inspection. |
| 9 | `print(data.head())` | Prints the first few rows so the dataset can be checked. |
| 10 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 3 Split Dataset` | Comment explaining this step: 3 Split Dataset. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `X = data.drop("Purchased", axis=1)` | Separates the input features by dropping the target column Purchased. |
| 4 | `y = data["Purchased"]` | Stores the target/output values that the model must predict. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `X_train, X_test, y_train, y_test = train_test_split(` | Splits features and target into training and testing datasets. |
| 7 | `    X, y,` | Passes the feature matrix and target vector to `train_test_split`. |
| 8 | `    test_size=0.25,` | Sets the fraction of data reserved for testing. |
| 9 | `    random_state=42` | Fixes randomness so results are reproducible. |
| 10 | `)` | Closes the multi-line statement or data structure started above. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `print("\nTraining size:", X_train.shape)` | Prints the shape/size to verify data dimensions. |
| 13 | `print("Testing size:", X_test.shape)` | Prints the shape/size to verify data dimensions. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 4 Build Decision Tree` | Comment explaining this step: 4 Build Decision Tree. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `dt_model = DecisionTreeClassifier(random_state=42)` | Creates a decision tree classifier. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `dt_model.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `train_pred = dt_model.predict(X_train)` | Generates predictions using the trained model. |
| 8 | `test_pred = dt_model.predict(X_test)` | Generates predictions using the trained model. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `print("\nDecision Tree Training Accuracy:",` | Prints the calculated accuracy value. |
| 11 | `      accuracy_score(y_train, train_pred))` | Calculates classification accuracy by comparing actual and predicted labels. |
| 12 | `` | Blank line used to separate code logically and improve readability. |
| 13 | `print("Decision Tree Testing Accuracy:",` | Prints the calculated accuracy value. |
| 14 | `      accuracy_score(y_test, test_pred))` | Calculates classification accuracy by comparing actual and predicted labels. |
| 15 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 5 Model Performance` | Comment explaining this step: 5 Model Performance. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("\nClassification Report:\n")` | Prints the given message or variable value for inspection. |
| 4 | `print(classification_report(y_test, test_pred))` | Prints the classification report with precision, recall, F1-score, and support. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 6 Cost Complexity Pruning` | Comment explaining this step: 6 Cost Complexity Pruning. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `path = dt_model.cost_complexity_pruning_path(X_train, y_train)` | Computes pruning alpha values for simplifying the decision tree. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `ccp_alphas = path.ccp_alphas` | Extracts the pruning alpha values from the pruning path. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `models = []` | Stores the computed value in \`models\` for use in later steps. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `for alpha in ccp_alphas:` | Extracts the pruning alpha values from the pruning path. |
| 10 | `    model = DecisionTreeClassifier(ccp_alpha=alpha)` | Creates a pruned decision tree using the current alpha value. |
| 11 | `    model.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |
| 12 | `    models.append(model)` | Adds the selected value or sequence to the list. |
| 13 | `` | Blank line used to separate code logically and improve readability. |
| 14 | `train_scores = [m.score(X_train, y_train) for m in models]` | Calculates model accuracy on the given dataset. |
| 15 | `test_scores = [m.score(X_test, y_test) for m in models]` | Calculates model accuracy on the given dataset. |
| 16 | `` | Blank line used to separate code logically and improve readability. |
| 17 | `plt.plot(ccp_alphas, train_scores, label="Train Accuracy")` | Plots values as a line graph for visualization. |
| 18 | `plt.plot(ccp_alphas, test_scores, label="Test Accuracy")` | Plots values as a line graph for visualization. |
| 19 | `plt.xlabel("Alpha")` | Labels the x-axis of the graph. |
| 20 | `plt.ylabel("Accuracy")` | Labels the y-axis of the graph. |
| 21 | `plt.legend()` | Displays the legend so plotted lines can be identified. |
| 22 | `plt.title("Cost Complexity Pruning")` | Adds a title to the current graph. |
| 23 | `plt.show()` | Displays the completed plot. |
| 24 | `` | Blank line used to separate code logically and improve readability. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 7 Random Forest` | Comment explaining this step: 7 Random Forest. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `rf_model = RandomForestClassifier(` | Creates a random forest ensemble classifier. |
| 4 | `    n_estimators=100,` | Sets how many estimators, such as trees or boosting stages, are used. |
| 5 | `    random_state=42` | Fixes randomness so results are reproducible. |
| 6 | `)` | Closes the multi-line statement or data structure started above. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `rf_model.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `rf_pred = rf_model.predict(X_test)` | Generates predictions using the trained model. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `print("\nRandom Forest Accuracy:",` | Prints the calculated accuracy value. |
| 13 | `      accuracy_score(y_test, rf_pred))` | Calculates classification accuracy by comparing actual and predicted labels. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# 8 AdaBoost with Decision Stumps` | Comment explaining this step: 8 AdaBoost with Decision Stumps. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `ada_model = AdaBoostClassifier(` | Creates an AdaBoost ensemble classifier. |
| 4 | `    n_estimators=50,` | Sets how many estimators, such as trees or boosting stages, are used. |
| 5 | `    random_state=42` | Fixes randomness so results are reproducible. |
| 6 | `)` | Closes the multi-line statement or data structure started above. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `ada_model.fit(X_train, y_train)` | Trains/fits the model or estimator using the provided data. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `ada_pred = ada_model.predict(X_test)` | Generates predictions using the trained model. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `print("\nAdaBoost Accuracy:",` | Prints the calculated accuracy value. |
| 13 | `      accuracy_score(y_test, ada_pred))` | Calculates classification accuracy by comparing actual and predicted labels. |

