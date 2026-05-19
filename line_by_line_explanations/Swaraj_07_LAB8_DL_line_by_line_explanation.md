# Swaraj_07_LAB8_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB8_DL.ipynb`

**Aim:** Convolutional Neural Network for Fashion-MNIST image classification with accuracy and loss visualization.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import tensorflow as tf` | Imports TensorFlow, the deep learning framework used to build and train neural networks. |
| 2 | `from tensorflow.keras import layers, models` | Imports layers for accessing Keras layer classes, models for accessing Keras model classes. |
| 3 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 4 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `fashion_mnist = tf.keras.datasets.fashion_mnist` | Stores the computed value in \`fashion_mnist\` for use in later steps. |
| 2 | `(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()` | Loads Fashion-MNIST training and testing data. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `# Normalize (0-255 -> 0-1)` | Comment explaining this step: Normalize (0-255 -> 0-1). |
| 2 | `x_train = x_train / 255.0` | Normalizes pixel values from 0-255 to 0-1. |
| 3 | `x_test = x_test / 255.0` | Normalizes pixel values from 0-255 to 0-1. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `# Reshape for CNN (add channel dimension)` | Comment explaining this step: Reshape for CNN (add channel dimension). |
| 6 | `x_train = x_train.reshape((60000, 28, 28, 1))` | Reshapes the array into the dimensions expected by the model. |
| 7 | `x_test = x_test.reshape((10000, 28, 28, 1))` | Reshapes the array into the dimensions expected by the model. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = models.Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# Convolution Layer 1` | Comment explaining this step: Convolution Layer 1. |
| 4 | `model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))` | Adds a convolution layer that learns image features using filters. |
| 5 | `model.add(layers.MaxPooling2D((2,2)))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `# Convolution Layer 2` | Comment explaining this step: Convolution Layer 2. |
| 8 | `model.add(layers.Conv2D(64, (3,3), activation='relu'))` | Adds a convolution layer that learns image features using filters. |
| 9 | `model.add(layers.MaxPooling2D((2,2)))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `# Flatten` | Comment explaining this step: Flatten. |
| 12 | `model.add(layers.Flatten())` | Flattens feature maps into a one-dimensional vector. |
| 13 | `` | Blank line used to separate code logically and improve readability. |
| 14 | `# Fully Connected Layer` | Comment explaining this step: Fully Connected Layer. |
| 15 | `model.add(layers.Dense(128, activation='relu'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 16 | `` | Blank line used to separate code logically and improve readability. |
| 17 | `# Output Layer (10 classes)` | Comment explaining this step: Output Layer (10 classes). |
| 18 | `model.add(layers.Dense(10, activation='softmax'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model.compile(optimizer='adam',` | Configures the model with optimizer, loss function, and metrics. |
| 2 | `              loss='sparse_categorical_crossentropy',` | Chooses the loss function the model minimizes. |
| 3 | `              metrics=['accuracy'])` | Chooses the evaluation metric reported during training and testing. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(x_train, y_train, epochs=10,` | Trains/fits the model or estimator using the provided data. |
| 2 | `                    validation_data=(x_test, y_test))` | Provides validation data to monitor performance during training. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `test_loss, test_acc = model.evaluate(x_test, y_test)` | Evaluates the trained model on validation or test data. |
| 2 | `print("Test Accuracy:", test_acc)` | Prints the calculated accuracy value. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `predictions = model.predict(x_test)` | Generates predictions using the trained model. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# Example prediction` | Comment explaining this step: Example prediction. |
| 4 | `print("Predicted:", np.argmax(predictions[0]))` | Prints the given message or variable value for inspection. |
| 5 | `print("Actual:", y_test[0])` | Prints the given message or variable value for inspection. |

## Code Cell 11

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.imshow(x_test[0].reshape(28,28), cmap='gray')` | Displays an image array, using grayscale when specified. |
| 2 | `plt.title(f"Predicted: {np.argmax(predictions[0])}")` | Adds a title to the current graph. |
| 3 | `plt.show()` | Displays the completed plot. |

## Code Cell 12

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['accuracy'])` | Plots values as a line graph for visualization. |
| 2 | `plt.plot(history.history['val_accuracy'])` | Plots values as a line graph for visualization. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `plt.title('Model Accuracy')` | Adds a title to the current graph. |
| 5 | `plt.xlabel('Epoch')` | Labels the x-axis of the graph. |
| 6 | `plt.ylabel('Accuracy')` | Labels the y-axis of the graph. |
| 7 | `plt.legend(['Train', 'Validation'])` | Displays the legend so plotted lines can be identified. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `plt.show()` | Displays the completed plot. |

## Code Cell 13

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['loss'])` | Plots values as a line graph for visualization. |
| 2 | `plt.plot(history.history['val_loss'])` | Plots values as a line graph for visualization. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `plt.title('Model Loss')` | Adds a title to the current graph. |
| 5 | `plt.xlabel('Epoch')` | Labels the x-axis of the graph. |
| 6 | `plt.ylabel('Loss')` | Labels the y-axis of the graph. |
| 7 | `plt.legend(['Train', 'Validation'])` | Displays the legend so plotted lines can be identified. |
| 8 | `` | Blank line used to separate code logically and improve readability. |
| 9 | `plt.show()` | Displays the completed plot. |

