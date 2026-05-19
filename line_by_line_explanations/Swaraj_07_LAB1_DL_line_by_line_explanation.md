# Swaraj_07_LAB1_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB1_DL.ipynb`

**Aim:** Principal Component Analysis (PCA) on a small numerical dataset, including standardization, explained variance, scree plots, and 2D projection.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 1

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import pandas as pd` | Imports Pandas for creating, loading, cleaning, and manipulating tabular datasets. |
| 3 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from sklearn.preprocessing import StandardScaler` | Imports StandardScaler for converting features to mean 0 and standard deviation 1. |
| 6 | `from sklearn.decomposition import PCA` | Imports PCA for dimensionality reduction and principal component analysis. |

## Code Cell 2

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# Sample numerical dataset` | Comment explaining this step: Sample numerical dataset. |
| 2 | `data = pd.DataFrame({` | Creates a Pandas DataFrame from the provided dictionary or table. |
| 3 | `    'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],` | Defines the values for the \`Feature1\` column in the DataFrame or output table. |
| 4 | `    'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],` | Defines the values for the \`Feature2\` column in the DataFrame or output table. |
| 5 | `    'Feature3': [1.1, 0.3, 1.5, 1.3, 1.8, 1.6, 1.0, 0.5, 1.2, 0.6]` | Defines the values for the \`Feature3\` column in the DataFrame or output table. |
| 6 | `})` | Closes the multi-line statement or data structure started above. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `print("Original Dataset:")` | Prints the given message or variable value for inspection. |
| 9 | `print(data)` | Prints the given message or variable value for inspection. |

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `scaler = StandardScaler()` | Creates a StandardScaler object for feature standardization. |
| 2 | `scaled_data = scaler.fit_transform(data)` | Fits the scaler on the data and transforms the values into scaled form. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `print("\nStandardized Data:")` | Prints the given message or variable value for inspection. |
| 5 | `print(scaled_data)` | Prints the given message or variable value for inspection. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `pca_full = PCA()` | Stores the computed value in \`pca_full\` for use in later steps. |
| 2 | `pca_full.fit(scaled_data)` | Fits PCA to learn principal components and their explained variance. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `explained_variance = pca_full.explained_variance_ratio_` | Stores how much variance each principal component explains. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `print("\nExplained Variance Ratio:")` | Prints the given message or variable value for inspection. |
| 7 | `print(explained_variance)` | Prints the given message or variable value for inspection. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure()` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `plt.plot(` | Plots values as a line graph for visualization. |
| 3 | `    range(1, len(explained_variance) + 1),` | Creates the sequence of x-axis values for the plot. |
| 4 | `    explained_variance,` | Passes this variable as an argument to the multi-line function call above. |
| 5 | `    marker='o'` | Uses circle markers on plotted points. |
| 6 | `)` | Closes the multi-line statement or data structure started above. |
| 7 | `plt.xlabel('Principal Components')` | Labels the x-axis of the graph. |
| 8 | `plt.ylabel('Explained Variance Ratio')` | Labels the y-axis of the graph. |
| 9 | `plt.title('Scree Plot (Individual Variance)')` | Adds a title to the current graph. |
| 10 | `plt.grid()` | Adds grid lines to make the graph easier to read. |
| 11 | `plt.show()` | Displays the completed plot. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure()` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `plt.plot(` | Plots values as a line graph for visualization. |
| 3 | `    np.cumsum(explained_variance),` | Calculates cumulative explained variance across principal components. |
| 4 | `    marker='o'` | Uses circle markers on plotted points. |
| 5 | `)` | Closes the multi-line statement or data structure started above. |
| 6 | `plt.xlabel('Number of Principal Components')` | Labels the x-axis of the graph. |
| 7 | `plt.ylabel('Cumulative Explained Variance')` | Labels the y-axis of the graph. |
| 8 | `plt.title('Scree Plot (Cumulative Variance)')` | Adds a title to the current graph. |
| 9 | `plt.grid()` | Adds grid lines to make the graph easier to read. |
| 10 | `plt.show()` | Displays the completed plot. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `pca = PCA(n_components=2)` | Stores the computed value in \`pca\` for use in later steps. |
| 2 | `pca_data = pca.fit_transform(scaled_data)` | Fits PCA and transforms the data into principal component values. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `print("\nData after PCA (2 Components):")` | Prints the given message or variable value for inspection. |
| 5 | `print(pca_data)` | Prints the given message or variable value for inspection. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure()` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `plt.scatter(pca_data[:, 0], pca_data[:, 1])` | Creates a scatter plot to show the relationship between two variables. |
| 3 | `plt.xlabel('Principal Component 1')` | Labels the x-axis of the graph. |
| 4 | `plt.ylabel('Principal Component 2')` | Labels the y-axis of the graph. |
| 5 | `plt.title('2D Visualization using PCA')` | Adds a title to the current graph. |
| 6 | `plt.grid()` | Adds grid lines to make the graph easier to read. |
| 7 | `plt.show()` | Displays the completed plot. |

A scree plot is a graph used mainly in PCA (Principal Component Analysis) and other dimensionality reduction methods to show how much variance each principal component explains.

It usually plots:

X-axis: component number, like PC1, PC2, PC3
Y-axis: eigenvalue or percentage of variance explained
The goal is to decide how many components to keep. You look for the “elbow” point, where the curve changes from steep to flat. Components before the elbow explain a lot of variance; components after it add only small amounts.

Example: if the plot drops sharply from PC1 to PC3 and then becomes almost flat, you might keep the first 3 principal components.

It is called a “scree” plot because the small, low-value components look like loose rubble at the bottom of a slope.