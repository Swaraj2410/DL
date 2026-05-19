# Swaraj_07_LAB6_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB6_DL.ipynb`

**Aim:** Convolutional Neural Network for MNIST handwritten digit classification, including preprocessing, training, and evaluation.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 3 | `import seaborn as sns` | Imports Seaborn for drawing a clearer heatmap of the confusion matrix. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from tensorflow.keras.datasets import mnist` | Imports mnist for loading handwritten digit images. |
| 6 | `from tensorflow.keras.models import Sequential` | Imports Sequential for creating layer-by-layer neural networks. |
| 7 | `from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout` | Imports Conv2D for convolutional image feature extraction, MaxPooling2D for downsampling feature maps, Flatten for converting feature maps into a vector, Dense for fully connected neural network layers, Dropout for reducing overfitting. |
| 8 | `from tensorflow.keras.utils import to_categorical` | Imports to_categorical for one-hot encoding class labels. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `from sklearn.metrics import confusion_matrix, classification_report` | Imports confusion_matrix for summarizing correct and wrong class predictions, classification_report for precision, recall, F1-score, and support. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `(X_train, y_train), (X_test, y_test) = mnist.load_data()` | Loads MNIST training and testing data. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("Training Data Shape:", X_train.shape)` | Prints the shape/size to verify data dimensions. |
| 4 | `print("Testing Data Shape:", X_test.shape)` | Prints the shape/size to verify data dimensions. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.figure(figsize=(10,5))` | Creates a new plotting figure, optionally with a specified size. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `for i in range(10):` | Starts a loop that repeats the indented block for each item in the range or sequence. |
| 4 | `    plt.subplot(2,5,i+1)` | Selects one subplot position inside a multi-plot figure. |
| 5 | `    plt.imshow(X_train[i], cmap='gray')` | Displays an image array, using grayscale when specified. |
| 6 | `    plt.title("Label: "+str(y_train[i]))` | Adds a title to the current graph. |
| 7 | `    plt.axis('off')` | Hides or configures plot axes; here it removes axes around images. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `plt.show()` | Displays the completed plot. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_train = X_train.reshape(60000,28,28,1)` | Reshapes the array into the dimensions expected by the model. |
| 2 | `X_test = X_test.reshape(10000,28,28,1)` | Reshapes the array into the dimensions expected by the model. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `X_train = X_train / 255.0` | Normalizes pixel values from 0-255 to 0-1. |
| 2 | `X_test = X_test / 255.0` | Normalizes pixel values from 0-255 to 0-1. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `y_train = to_categorical(y_train,10)` | Converts integer labels into one-hot encoded vectors. |
| 2 | `y_test = to_categorical(y_test,10)` | Converts integer labels into one-hot encoded vectors. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# First Convolution Layer` | Comment explaining this step: First Convolution Layer. |
| 4 | `model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))` | Adds a convolution layer that learns image features using filters. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `# Pooling Layer` | Comment explaining this step: Pooling Layer. |
| 7 | `model.add(MaxPooling2D(pool_size=(2,2)))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `# Second Convolution Layer` | Comment explaining this step: Second Convolution Layer. |
| 10 | `model.add(Conv2D(64,(3,3),activation='relu'))` | Adds a convolution layer that learns image features using filters. |
| 11 | `` | Blank line used to separate code logically and improve readability. |
| 12 | `# Pooling Layer` | Comment explaining this step: Pooling Layer. |
| 13 | `model.add(MaxPooling2D(pool_size=(2,2)))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 14 | `` | Blank line used to separate code logically and improve readability. |
| 15 | `# Flatten Layer` | Comment explaining this step: Flatten Layer. |
| 16 | `model.add(Flatten())` | Flattens feature maps into a one-dimensional vector. |
| 17 | `` | Blank line used to separate code logically and improve readability. |
| 18 | `# Fully Connected Layer` | Comment explaining this step: Fully Connected Layer. |
| 19 | `model.add(Dense(128,activation='relu'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 20 | `` | Blank line used to separate code logically and improve readability. |
| 21 | `# Dropout Layer (prevents overfitting)` | Comment explaining this step: Dropout Layer (prevents overfitting). |
| 22 | `model.add(Dropout(0.5))` | Adds dropout regularization to reduce overfitting. |
| 23 | `` | Blank line used to separate code logically and improve readability. |
| 24 | `# Output Layer (10 classes)` | Comment explaining this step: Output Layer (10 classes). |
| 25 | `model.add(Dense(10,activation='softmax'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model.compile(` | Configures the model with optimizer, loss function, and metrics. |
| 2 | `    optimizer='adam',` | Chooses the algorithm used to update model weights. |
| 3 | `    loss='categorical_crossentropy',` | Chooses the loss function the model minimizes. |
| 4 | `    metrics=['accuracy']` | Chooses the evaluation metric reported during training and testing. |
| 5 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model.summary()` | Prints the neural network architecture and parameter counts. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(` | Trains/fits the model or estimator using the provided data. |
| 2 | `    X_train,` | Passes this variable as an argument to the multi-line function call above. |
| 3 | `    y_train,` | Passes this variable as an argument to the multi-line function call above. |
| 4 | `    epochs=10,` | Sets the number of complete training passes through the dataset. |
| 5 | `    batch_size=64,` | Sets how many samples are processed before each weight update. |
| 6 | `    validation_split=0.2` | Reserves a percentage of images for validation. |
| 7 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `test_loss, test_accuracy = model.evaluate(X_test,y_test)` | Evaluates the trained model on validation or test data. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `print("Test Accuracy:",test_accuracy)` | Prints the calculated accuracy value. |

## Code Cell 14

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['accuracy'])` | Plots values as a line graph for visualization. |
| 2 | `plt.plot(history.history['val_accuracy'])` | Plots values as a line graph for visualization. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `plt.title("Model Accuracy")` | Adds a title to the current graph. |
| 5 | `plt.xlabel("Epoch")` | Labels the x-axis of the graph. |
| 6 | `plt.ylabel("Accuracy")` | Labels the y-axis of the graph. |
| 7 | `plt.legend(['Train','Validation'])` | Displays the legend so plotted lines can be identified. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `plt.show()` | Displays the completed plot. |

## Code Cell 15

| Line | Code | Explanation |
|---:|---|---|
| 1 | `y_pred = model.predict(X_test)` | Generates predictions using the trained model. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# Convert probabilities to class labels` | Comment explaining this step: Convert probabilities to class labels. |
| 4 | `y_pred_classes = np.argmax(y_pred,axis=1)` | Chooses the class index with the highest predicted probability. |
| 5 | `y_true = np.argmax(y_test,axis=1)` | Chooses the class index with the highest predicted probability. |

## Code Cell 16

| Line | Code | Explanation |
|---:|---|---|
| 1 | `cm = confusion_matrix(y_true,y_pred_classes)` | Creates a confusion matrix from true and predicted labels. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `plt.figure(figsize=(10,8))` | Creates a new plotting figure, optionally with a specified size. |
| 4 | `sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')` | Draws the confusion matrix as an annotated heatmap. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `plt.xlabel("Predicted Label")` | Labels the x-axis of the graph. |
| 7 | `plt.ylabel("True Label")` | Labels the y-axis of the graph. |
| 8 | `plt.title("Confusion Matrix")` | Adds a title to the current graph. |
| 9 | `` | Blank line used to separate code logically and improve readability. |
| 10 | `plt.show()` | Displays the completed plot. |

## Code Cell 17

| Line | Code | Explanation |
|---:|---|---|
| 1 | `print(classification_report(y_true,y_pred_classes))` | Prints the classification report with precision, recall, F1-score, and support. |

