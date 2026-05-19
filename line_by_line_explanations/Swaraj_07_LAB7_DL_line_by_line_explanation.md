# Swaraj_07_LAB7_DL - Line by Line Code Explanation

**Notebook:** `Swaraj_07_LAB7_DL.ipynb`

**Aim:** Convolutional Neural Network for plant disease image classification using a Kaggle PlantVillage dataset.

This document is written for examiner/viva preparation. Each code line is listed with its role in the program.

## Code Cell 3

| Line | Code | Explanation |
|---:|---|---|
| 1 | `` | Blank line used to separate code logically and improve readability. |
| 2 | `!pip install kaggle` | Installs the required Python package(s): kaggle. |
| 3 | `` | Blank line used to separate code logically and improve readability. |
| 4 | `# Upload kaggle.json (your API key)` | Comment explaining this step: Upload kaggle.json (your API key). |
| 5 | `from google.colab import files` | Imports files for uploading local files into Google Colab. |
| 6 | `files.upload()` | Opens the Colab upload dialog so `kaggle.json` can be uploaded. |
| 7 | `` | Blank line used to separate code logically and improve readability. |
| 8 | `# Setup` | Comment explaining this step: Setup. |
| 9 | `!mkdir -p ~/.kaggle` | Creates the Kaggle configuration folder in the Colab/Linux environment. |
| 10 | `!cp kaggle.json ~/.kaggle/` | Copies kaggle.json into the Kaggle configuration folder. |
| 11 | `!chmod 600 ~/.kaggle/kaggle.json` | Sets secure permissions for kaggle.json so Kaggle accepts the API key. |
| 12 | `` | Blank line used to separate code logically and improve readability. |
| 13 | `# Download dataset` | Comment explaining this step: Download dataset. |
| 14 | `!kaggle datasets download -d emmarex/plantdisease` | Downloads the selected PlantVillage dataset from Kaggle. |
| 15 | `` | Blank line used to separate code logically and improve readability. |
| 16 | `# Unzip` | Comment explaining this step: Unzip. |
| 17 | `!unzip plantdisease.zip` | Extracts the downloaded zip file into usable folders. |

## Code Cell 4

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import numpy as np` | Imports NumPy for numerical arrays, reshaping, mathematical operations, and model input preparation. |
| 2 | `import matplotlib.pyplot as plt` | Imports Matplotlib pyplot for plotting graphs and visualizing data or model performance. |
| 3 | `import tensorflow as tf` | Imports TensorFlow, the deep learning framework used to build and train neural networks. |
| 4 | `` | Blank line used to separate code logically and improve readability. |
| 5 | `from tensorflow.keras.preprocessing.image import ImageDataGenerator` | Imports ImageDataGenerator for loading and preprocessing images from folders. |
| 6 | `from tensorflow.keras.models import Sequential` | Imports Sequential for creating layer-by-layer neural networks. |
| 7 | `from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout` | Imports Conv2D for convolutional image feature extraction, MaxPooling2D for downsampling feature maps, Flatten for converting feature maps into a vector, Dense for fully connected neural network layers, Dropout for reducing overfitting. |

## Code Cell 5

| Line | Code | Explanation |
|---:|---|---|
| 1 | `train_dir = "PlantVillage"` | Stores the computed value in \`train_dir\` for use in later steps. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `img_size = 128` | Stores an important hyperparameter used later in the notebook. |
| 4 | `batch_size = 32` | Sets how many samples are processed before each weight update. |
| 5 | `` | Blank line used to separate code logically and improve readability. |
| 6 | `datagen = ImageDataGenerator(` | Creates an image generator that rescales images and reserves validation data. |
| 7 | `    rescale=1./255,` | Scales image pixel values from 0-255 to 0-1. |
| 8 | `    validation_split=0.2` | Reserves a percentage of images for validation. |
| 9 | `)` | Closes the multi-line statement or data structure started above. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `train_data = datagen.flow_from_directory(` | Loads labeled images from class subfolders in batches. |
| 12 | `    train_dir,` | Passes this variable as an argument to the multi-line function call above. |
| 13 | `    target_size=(img_size, img_size),` | Resizes every input image to the required height and width. |
| 14 | `    batch_size=batch_size,` | Sets how many samples are processed before each weight update. |
| 15 | `    class_mode='categorical',` | Specifies categorical labels for multi-class image classification. |
| 16 | `    subset='training'` | Selects whether the generator reads the training or validation split. |
| 17 | `)` | Closes the multi-line statement or data structure started above. |
| 18 | `` | Blank line used to separate code logically and improve readability. |
| 19 | `val_data = datagen.flow_from_directory(` | Loads labeled images from class subfolders in batches. |
| 20 | `    train_dir,` | Passes this variable as an argument to the multi-line function call above. |
| 21 | `    target_size=(img_size, img_size),` | Resizes every input image to the required height and width. |
| 22 | `    batch_size=batch_size,` | Sets how many samples are processed before each weight update. |
| 23 | `    class_mode='categorical',` | Specifies categorical labels for multi-class image classification. |
| 24 | `    subset='validation'` | Selects whether the generator reads the training or validation split. |
| 25 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 6

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model = Sequential()` | Creates a sequential neural network where layers are added one by one. |
| 2 | `` | Blank line used to separate code logically and improve readability. |
| 3 | `# Convolution Block 1` | Comment explaining this step: Convolution Block 1. |
| 4 | `model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))` | Adds a convolution layer that learns image features using filters. |
| 5 | `model.add(MaxPooling2D(2,2))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 6 | `` | Blank line used to separate code logically and improve readability. |
| 7 | `# Convolution Block 2` | Comment explaining this step: Convolution Block 2. |
| 8 | `model.add(Conv2D(64, (3,3), activation='relu'))` | Adds a convolution layer that learns image features using filters. |
| 9 | `model.add(MaxPooling2D(2,2))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 10 | `` | Blank line used to separate code logically and improve readability. |
| 11 | `# Convolution Block 3` | Comment explaining this step: Convolution Block 3. |
| 12 | `model.add(Conv2D(128, (3,3), activation='relu'))` | Adds a convolution layer that learns image features using filters. |
| 13 | `model.add(MaxPooling2D(2,2))` | Adds a max-pooling layer to reduce spatial size while keeping important features. |
| 14 | `` | Blank line used to separate code logically and improve readability. |
| 15 | `# Flatten` | Comment explaining this step: Flatten. |
| 16 | `model.add(Flatten())` | Flattens feature maps into a one-dimensional vector. |
| 17 | `` | Blank line used to separate code logically and improve readability. |
| 18 | `# Dense Layers` | Comment explaining this step: Dense Layers. |
| 19 | `model.add(Dense(128, activation='relu'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |
| 20 | `model.add(Dropout(0.5))` | Adds dropout regularization to reduce overfitting. |
| 21 | `model.add(Dense(train_data.num_classes, activation='softmax'))` | Adds a fully connected neural network layer; in regression/time-series models, one unit produces one numeric output. |

## Code Cell 7

| Line | Code | Explanation |
|---:|---|---|
| 1 | `model.compile(` | Configures the model with optimizer, loss function, and metrics. |
| 2 | `    optimizer='adam',` | Chooses the algorithm used to update model weights. |
| 3 | `    loss='categorical_crossentropy',` | Chooses the loss function the model minimizes. |
| 4 | `    metrics=['accuracy']` | Chooses the evaluation metric reported during training and testing. |
| 5 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 8

| Line | Code | Explanation |
|---:|---|---|
| 1 | `history = model.fit(` | Trains/fits the model or estimator using the provided data. |
| 2 | `    train_data,` | Passes this variable as an argument to the multi-line function call above. |
| 3 | `    validation_data=val_data,` | Provides validation data to monitor performance during training. |
| 4 | `    epochs=7` | Sets the number of complete training passes through the dataset. |
| 5 | `)` | Closes the multi-line statement or data structure started above. |

## Code Cell 9

| Line | Code | Explanation |
|---:|---|---|
| 1 | `loss, acc = model.evaluate(val_data)` | Evaluates the trained model on validation or test data. |
| 2 | `print("Validation Accuracy:", acc)` | Prints the calculated accuracy value. |

## Code Cell 10

| Line | Code | Explanation |
|---:|---|---|
| 1 | `plt.plot(history.history['accuracy'])` | Plots values as a line graph for visualization. |
| 2 | `plt.plot(history.history['val_accuracy'])` | Plots values as a line graph for visualization. |
| 3 | `plt.title('Model Accuracy')` | Adds a title to the current graph. |
| 4 | `plt.xlabel('Epoch')` | Labels the x-axis of the graph. |
| 5 | `plt.ylabel('Accuracy')` | Labels the y-axis of the graph. |
| 6 | `plt.legend(['Train', 'Validation'])` | Displays the legend so plotted lines can be identified. |
| 7 | `plt.show()` | Displays the completed plot. |

