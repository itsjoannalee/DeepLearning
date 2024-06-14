#!/usr/bin/env python
# coding: utf-8

# # 1. Using Convolutional Neural Network for Image Recognition
# 
# 

# In[32]:


import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import utils, datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import pickle
import csv
import itertools
from collections import defaultdict
import time
import pandas as pd
import math
from tqdm import tqdm
get_ipython().system('pip install dill')
import dill


# In[43]:


tf.test.is_gpu_available()


# In[34]:


tf.config.list_physical_devices('GPU')


# In[35]:


tf.__version__


# In[333]:


import urllib.request
if not os.path.exists("lab11_1_lib.py"):
    urllib.request.urlretrieve("https://nthu-datalab.github.io/ml/labs/11-1_CNN/lab11_1_lib.py", "lab11_1_lib.py")

from lab11_1_lib import draw_timeline


# ### import MNIST dataset from Tensorflow.
# 

# In[330]:


# Download and prepare the MNIST dataset
(train_image, train_label), (test_image, test_label) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_image, test_image = train_image / 255.0, test_image / 255.0
print('shape of train_image:', train_image.shape)
print('shape of train_label:', train_label.shape)
print('shape of test_image:', test_image.shape)
print('shape of test_label:', test_label.shape)


# ## Multilayer Convolutional Network on MNIST

# In[331]:


# reshaping the training data to 3 dimensions
train_image_2 = train_image.reshape((60000, 28, 28, 1))
test_image_2 = test_image.reshape((10000, 28, 28, 1))
print(train_image_2.shape)
print(test_image_2.shape)


# MNIST has one color channel.(because the image are grayscale)\
# In this example, we will configure our CNN to process inputs of shape (28, 28, 1), which is the format of MNIST image. We do this by passing the argument input_shape to our first layer.

# # 1-1
# Please implement a CNN for image recognition using the MNIST dataset. You have to design
# your network architecture and analyze the effect of different stride size and filter size. Also,
# plot the learning curve, accuracy of training and test sets, and distributions of weights and
# biases.

# ## model_1

# data spilt: 55, 000 examples are in the training set, 5, 000 in the validation
# set, and 10, 000 in the test set.
# model setting:
# 
# * Epochs = 100
# * batch size = 5000
# * strides=(1,1),filter=(3,3)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[332]:


#The convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model_1 = models.Sequential()
model_1.add(layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_1.add(layers.MaxPooling2D((2, 2)))
model_1.add(layers.Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'))
model_1.add(layers.MaxPooling2D((2, 2)))
model_1.add(layers.Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'))
model_1.add(layers.Flatten())
model_1.add(layers.Dense(64, activation='relu'))
model_1.add(layers.Dropout(0.5))
model_1.add(layers.Dense(10, activation='softmax'))
model_1.summary()


# In[334]:


model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[335]:


from tensorflow.keras.callbacks import Callback

# 回调函数定义
class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback = TestAccuracyCallback(test_data=(test_image_2, test_label))


# In[336]:


np.random.seed(111024520)
history = model_1.fit(train_image_2, train_label , batch_size=5000, epochs=100, validation_split=1/12, callbacks=[test_callback])


# In[337]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.15,7/3))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights = []

for layer in model_1.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 3:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights)

plt.show()


# From the above plots, we can find that:
# 
# 
# 1.   After 20 epochs, the accuracy rates and learning rate tend to stabilize.
# 2.   The distribution of the weights in layer2,3, dense1 looks like a normal distribution.
# 
# 
# 

# ## model_2

# Then, we try to set different stride size$=(2,2).$

# * Epochs = 100
# * batch size = 5000
# * strides=(2,2),filter=(3,3)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[338]:


#The convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model_2 = models.Sequential()
model_2.add(layers.Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(layers.Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(layers.Conv2D(64, (3, 3), strides=(2,2), padding='same', activation='relu'))
model_2.add(layers.Flatten())
model_2.add(layers.Dense(64, activation='relu'))
model_2.add(layers.Dropout(0.5))
model_2.add(layers.Dense(10, activation='softmax'))
model_2.summary()


# In[339]:


model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[340]:


from tensorflow.keras.callbacks import Callback

# 回调函数定义
class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback2 = TestAccuracyCallback(test_data=(test_image_2, test_label))


# In[341]:


np.random.seed(111024520)

history_2 = model_2.fit(train_image_2, train_label, batch_size=5000, epochs=100, validation_split=1/12, callbacks=[test_callback2])


# In[342]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.1,7/3))
plt.subplot(1, 2, 1)
plt.plot(history_2.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history_2.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback2.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history_2.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights_2 = []

for layer in model_2.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights_2.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 3:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights_2)

plt.show()


# From the above plot, we can find that when we increse thestride size from (1,1) to (2,2), the results are similar to the previous model.\
# However, this model has fewer trainable parameters compared to the previous one, resulting in lower computational costs.

# ## model_3

# Then, we try to set different filter size$=(5,5).$

# * Epochs = 100
# * batch size = 5000
# * strides=(1,1),filter=(5,5)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[343]:


#The convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model_3 = models.Sequential()
model_3.add(layers.Conv2D(32, (5, 5), strides=(1,1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model_3.add(layers.MaxPooling2D((2, 2)))
model_3.add(layers.Conv2D(64, (5, 5), strides=(1,1), padding='same', activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))
model_3.add(layers.Conv2D(64, (5, 5), strides=(1,1), padding='same', activation='relu'))
model_3.add(layers.Flatten())
model_3.add(layers.Dense(64, activation='relu'))
model_3.add(layers.Dropout(0.5))
model_3.add(layers.Dense(10, activation='softmax'))
model_3.summary()


# In[344]:


model_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[345]:


from tensorflow.keras.callbacks import Callback

# 回调函数定义
class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback3 = TestAccuracyCallback(test_data=(test_image_2, test_label))


# In[346]:


np.random.seed(111024520)

history_3 = model_3.fit(train_image_2, train_label, batch_size=5000, epochs=100, validation_split=1/12, callbacks=[test_callback3])


# In[347]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(8.97,7/3))
plt.subplot(1, 2, 1)
plt.plot(history_3.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history_3.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback3.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history_3.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights_3 = []

for layer in model_3.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights_3.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 3:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights_3)

plt.show()


# From the above plot, we can find that when we increse the
# filter size from (3,3) to (5,5), the results are similar to the previous model.\
# However, this model has more trainable parameters compared to the model 1, resulting in higher computational costs.

# # 1-2 Show some examples of correctly classified and miss-classified images and discuss your results.
# 

# In[348]:


predictions = model_1.predict(test_image_2)
predicted_labels = np.argmax(predictions, axis=1) #10000個

# true_and_pred_4
true_and_pred_4_indices = np.where((test_label == 4) & (predicted_labels == 4))[0]

# true_not_4_pred_4
true_not_4_pred_4_indices = np.where((test_label != 4) & (predicted_labels == 4))[0]

# true_and_pred_4 第一張圖
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(test_image[true_and_pred_4_indices[0]], cmap='gray')
plt.title(f'True: {test_label[true_and_pred_4_indices[0]]}\nPred: {predicted_labels[true_and_pred_4_indices[0]]}')
plt.axis('off')

# true_not_4_pred_4 第一張圖
plt.subplot(1, 2, 2)
plt.imshow(test_image[true_not_4_pred_4_indices[0]], cmap='gray')
plt.title(f'True: {test_label[true_not_4_pred_4_indices[0]]}\nPred: {predicted_labels[true_not_4_pred_4_indices[0]]}')
plt.axis('off')

plt.tight_layout()
plt.show()


# The above plots are examples of correctly classified and mis-classified images.\
# It show a true label and predicted label of 4 and another example where the true label is not 4 but the predicted label is 4.\
# I think the pattern of right image looks like 4, so it was miss-classified.

# # 1-3 Following 1-2, observe the feature maps from different convolutional layers and describe how a feature map changes with increasing depth.

# In[349]:


# Choose an input image from the test set
input_image_1 = test_image_2[true_and_pred_4_indices[0]].reshape(1, 28, 28, 1)
input_image_2 = test_image_2[true_not_4_pred_4_indices[0]].reshape(1, 28, 28, 1)

# Get feature maps for the last convolutional layer in each block
ixs = [0, 2, 4]
outputs = [model_1.layers[i].output for i in ixs]
model = tf.keras.Model(inputs=model_1.input, outputs=outputs)
feature_maps_1 = model.predict(input_image_1)
feature_maps_2 = model.predict(input_image_2)
layer_names = [layer.name for layer in model_1.layers[:5] if 'conv2d' in layer.name ]

# Plot the feature maps
plt.figure(figsize=(16, 4))

# Plot the original images for both cases
plt.subplot(2, 4, 1)
plt.imshow(test_image[true_and_pred_4_indices[0]], cmap='gray')
plt.title(f'True: {test_label[true_and_pred_4_indices[0]]}\nPred: {predicted_labels[true_and_pred_4_indices[0]]}')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(test_image[true_not_4_pred_4_indices[0]], cmap='gray')
plt.title(f'\nTrue: {test_label[true_not_4_pred_4_indices[0]]}\nPred: {predicted_labels[true_not_4_pred_4_indices[0]]}')
plt.axis('off')

j=1
# Plot the feature maps for each block for the first case
for i, fmap in enumerate(feature_maps_1):
    plt.subplot(2, 4, i + 2)  # i + 2 because the original image is at position 1
    plt.imshow(fmap[0, :, :, 0], cmap='gray')  # Assuming the last dimension represents the number of filters
    plt.title(f'Feature Map - conv_{j}')
    plt.axis('off')

j=1
# Plot the feature maps for each block for the second case
for i, fmap in enumerate(feature_maps_2):
    plt.subplot(2, 4, i + 6)  # i + 6 because the original image is at position 5
    plt.imshow(fmap[0, :, :, 0], cmap='gray')  # Assuming the last dimension represents the number of filters
    plt.title(f'Feature Map - conv_{j}')
    j+=1
    plt.axis('off')

# Adjust the separation between rows
plt.subplots_adjust(hspace=0.5)
# Show the figure
plt.show()


# From the above plot, we can find that:
# 
# 1.   Feature maps closer to the input of the model capture a lot of fine detail in the image and that as we progress deeper into the model, the feature maps show less and less detail.
# 
# 2.   This pattern was to be expected, as the model abstracts the features from the image into more general concepts that can be used to make a classification.
# 
# 

# # 1-4 Following 1-1, please add L2 regularization to the CNN implemented in 1-1 and discuss its effect.
# 

# We set $α=0.05$ of L2 regularization.

# * Epochs = 100
# * batch size = 5000
# * strides=(1,1),filter=(3,3)
# * kernel_regularizer=0.05
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[350]:


from tensorflow.keras import layers, models, regularizers

# Build the CNN model with L2 regularization
model_L2 = models.Sequential()
model_L2.add(layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.05), input_shape=(28, 28, 1)))
model_L2.add(layers.MaxPooling2D((2, 2)))
model_L2.add(layers.Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.05)))
model_L2.add(layers.MaxPooling2D((2, 2)))
model_L2.add(layers.Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.05)))

model_L2.add(layers.Flatten())
model_L2.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
model_L2.add(layers.Dropout(0.5))
model_L2.add(layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.05)))
model_L2.summary()
# Compile the model
model_L2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


# In[351]:


from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback4 = TestAccuracyCallback(test_data=(test_image_2, test_label))


# In[352]:


np.random.seed(111024520)

history_L2 = model_L2.fit(train_image_2, train_label, batch_size=5000, epochs=100, validation_split=1/12, callbacks=[test_callback4])


# In[353]:


_, test_acc_L2 = model_L2.evaluate(test_image_2, test_label, verbose=0)
print('Testing Accuracy : %.4f'%test_acc_L2)


# In[354]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.1,7/3))
plt.subplot(1, 2, 1)
plt.plot(history_L2.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history_L2.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback4.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history_L2.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights_4 = []

for layer in model_L2.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights_4.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 3:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights_4)

plt.show()


# We can find that:
# 
# 
# 1.  The histograms of weights  appear to be concentrated around a narrow range, and the value appear to 0.
# 
# 2.  The regularization term encourages the network to learn smaller weight values.
# 

# # 2 Preprocessing Before Using Convolutional Neural Network for Image Recognition

# # 2-5 Data preprocessing
# * data: a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# 
# * labels: a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
# 
# 1. We first reshape the batches of data to be $(10000,3,32,32)$, and then combined 5 batches to be a train set image with dim$=(50000, 32, 32, 3)$.
# 2. Similarly, we reshape the batches of label and combined them to be to be a train set label with dim$=(50000, 1)$
# 3. Reshape test set image to be dim$=(10000,3,32,32)$.
# 4. Reshape test set label to be dim$=(10000,1)$.

# In[85]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[90]:


batch = unpickle("C:/Users/cluster/Desktop/lee/HW2/batches.meta")


# In[236]:


label_names = batch[b'label_names']
label_names = [label.decode('utf-8') for label in label_names]
label_names


# In[91]:


data1 = unpickle("C:/Users/cluster/Desktop/lee/HW2/data_batch_1")
data1, label1 = data1[b'data'], data1[b'labels'] #10000x3072, 10000
data2 = unpickle("C:/Users/cluster/Desktop/lee/HW2/data_batch_2")
data2, label2 = data2[b'data'], data2[b'labels']
data3 = unpickle("C:/Users/cluster/Desktop/lee/HW2/data_batch_3")
data3, label3 = data3[b'data'], data3[b'labels']
data4 = unpickle("C:/Users/cluster/Desktop/lee/HW2/data_batch_4")
data4, label4 = data4[b'data'], data4[b'labels']
data5 = unpickle("C:/Users/cluster/Desktop/lee/HW2/data_batch_5")
data5, label5 = data5[b'data'], data5[b'labels']


# In[194]:


data1_new = data1.reshape(10000,3,32,32) #每個圖像的通道維度（紅、綠、藍）被分開。
data2_new = data2.reshape(10000,3,32,32)
data3_new = data3.reshape(10000,3,32,32)
data4_new = data4.reshape(10000,3,32,32)
data5_new = data5.reshape(10000,3,32,32)


# In[195]:


train_image2 = np.concatenate((data1_new, data2_new, data3_new, data4_new, data5_new), axis=0)
train_image2=np.transpose(train_image2, (0,2,3,1))
print(train_image2.shape)


# In[204]:


#label1是list轉成array
train_label2 = np.concatenate((np.array(label1),np.array(label2),np.array(label3),np.array(label4),np.array(label5)))
train_label2=train_label2.reshape(50000,)


# In[205]:


data_test = unpickle("C:/Users/cluster/Desktop/lee/HW2/test_batch")
test_image2, test_label2 = data_test[b'data'], data_test[b'labels']


# In[206]:


test_image2 = test_image2.reshape(10000,3,32,32)
test_image2 = np.transpose(test_image2, (0,2,3,1))
test_label2 = (np.array(test_label2)).reshape(10000,)


# In[232]:


test_label2


# In[207]:


print('X_train shape:', train_image2.shape)
print('Y_train shape:', train_label2.shape)
print('X_test shape:', test_image2.shape)
print('Y_test shape:', test_label2.shape)


# In[237]:


plt.figure(figsize=(6, 3))
plt.imshow(train_image2[150], cmap='gray')
plt.title(f'True: {label_names[train_label2[150]]}')
plt.axis('off')
plt.xticks([])
plt.show()


# 
# 
# # 2-1 CNN on CIFAR-10

# ## model 2_1

# * Epochs = 200
# * batch size = 500
# * strides=(3,3),filter=(5,5)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[120]:


model2_1 = models.Sequential()

#The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model2_1.add(layers.Conv2D(64, (5, 5), strides=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model2_1.add(layers.MaxPool2D(((2, 2))))
model2_1.add(layers.Conv2D(64, (5, 5), strides=(3,3),padding='same', activation='relu'))
model2_1.add(layers.MaxPool2D(((2, 2))))

model2_1.add(layers.Flatten())

model2_1.add(layers.Dense(384, activation='relu'))
model2_1.add(layers.Dropout(0.2))
model2_1.add(layers.Dense(192, activation='relu'))
model2_1.add(layers.Dropout(0.2))
model2_1.add(layers.Dense(10, activation='softmax'))
model2_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2_1.summary()


# In[121]:


from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback2_1 = TestAccuracyCallback(test_data=(test_image2, test_label2))


# In[122]:


history2_1 = model2_1.fit(train_image2, train_label2, batch_size=500, epochs=200, validation_split=5000/50000, callbacks=[test_callback2_1])


# In[123]:


model2_1.layers


# In[124]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.15,7/3))
plt.subplot(1, 2, 1)
plt.plot(history2_1.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history2_1.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback2_1.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history2_1.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights2_1 = []

for layer in model2_1.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights2_1.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 2:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 3:
            axes[row, col].set_title(f'Histogram of Dense 2')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights2_1)

plt.show()


# From the above plots, we can find that:
# 
# 
# 1.   After 50 epochs, the validation accuracy and testing accuracy rates tend to stabilize around 0.5.
# 2.   The learning rate decreases slowly.
# 3.   The distribution of the weights in layers looks like a normal distribution.
# 
# 
# 

# ## model2_2

# Then, we try to set different stride size$=(1,1).$

# * Epochs = 200
# * batch size = 500
# * strides=(1,1),filter=(5,5)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[130]:


model2_2 = models.Sequential()

#The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model2_2.add(layers.Conv2D(64, (5, 5), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))
model2_2.add(layers.MaxPool2D(((2, 2))))
model2_2.add(layers.Conv2D(64, (5, 5), strides=(1,1),padding='same', activation='relu'))
model2_2.add(layers.MaxPool2D(((2, 2))))

model2_2.add(layers.Flatten())
model2_2.add(layers.Dense(384, activation='relu'))
model2_2.add(layers.Dropout(0.2))
model2_2.add(layers.Dense(192, activation='relu'))
model2_2.add(layers.Dropout(0.2))
model2_2.add(layers.Dense(10, activation='softmax'))
model2_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2_2.summary()


# In[131]:


from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback2_2 = TestAccuracyCallback(test_data=(test_image2, test_label2))


# In[132]:


history2_2 = model2_2.fit(train_image2, train_label2, batch_size=500, epochs=200, validation_split=5000/50000, callbacks=[test_callback2_2])


# In[134]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.15,7/3))
plt.subplot(1, 2, 1)
plt.plot(history2_2.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history2_2.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback2_2.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history2_2.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights2_2 = []

for layer in model2_2.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights2_2.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 2:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 3:
            axes[row, col].set_title(f'Histogram of Dense 2')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])
        
    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights2_2)

plt.show()


# From the above plot, we can find that when we increse thestride size from (3,3) to (1,1), the results are similar to the previous model.
#  1. After 100 epochs, the learning rate tend to stabilize. 
#  2. After 50 epochs,the validation accuracy and testing accuracy rates tend to stabilize around 0.5.
#  
# However, this model has more trainable parameters compared to the previous one, resulting in higher computational costs.

# ## model2_3

# Then, we try to set different filter size$=(2,2).$

# * Epochs = 200
# * batch size = 500
# * strides=(3,3),filter=(2,2)
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[135]:


model2_3 = models.Sequential()

#The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model2_3.add(layers.Conv2D(64, (2, 2), strides=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model2_3.add(layers.MaxPool2D(((2, 2))))
model2_3.add(layers.Conv2D(64, (2, 2), strides=(3,3),padding='same', activation='relu'))
model2_3.add(layers.MaxPool2D(((2, 2))))

model2_3.add(layers.Flatten())
model2_3.add(layers.Dense(384, activation='relu'))
model2_3.add(layers.Dropout(0.2))
model2_3.add(layers.Dense(192, activation='relu'))
model2_3.add(layers.Dropout(0.2))
model2_3.add(layers.Dense(10, activation='softmax'))
model2_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2_3.summary()


# In[136]:


from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback2_3 = TestAccuracyCallback(test_data=(test_image2, test_label2))


# In[137]:


history2_3 = model2_3.fit(train_image2, train_label2, batch_size=500, epochs=200, validation_split=5000/50000, callbacks=[test_callback2_3])


# In[139]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.15,7/3))
plt.subplot(1, 2, 1)
plt.plot(history2_3.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history2_3.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback2_3.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history2_3.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights2_3 = []

for layer in model2_3.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights2_3.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 2:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 3:
            axes[row, col].set_title(f'Histogram of Dense 2')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')
        
    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights2_3)

plt.show()


# From the above plot, we can find that when we increse the
# filter size from (5,5) to (2,2), the results are similar to the previous model.
# 
# 1. After 50 epochs, the validation accuracy and testing accuracy rates tend to stabilize around 0.4.
# 2. The learning rate decreases slowly.
# 
# However, this model has less trainable parameters compared to the model 2_1, resulting in lower computational costs.

# # 2-2 Show some examples of correctly classified and miss-classified images and discuss your results.
# 

# In[238]:


pred = model2_1.predict(test_image2)
pred_labels = np.argmax(pred, axis=1) #10000個

# true_and_pred_cat
correct_indices = np.where((test_label2 == 3) & (pred_labels == 3))[0]

# true_not_cat_pred_cat
wrong_indices = np.where((test_label2 != 3) & (pred_labels == 3))[0]

# true_and_pred_cat 第一張圖
plt.figure(figsize=(6, 3))
plot_label = plt.GridSpec(1, 2, wspace=0.5)
plt.subplot(1, 2, 1)
plt.imshow(test_image2[correct_indices[0]], cmap=plt.get_cmap('gray'))
plt.title(f'True: {label_names[test_label2[correct_indices[0]]]}\nPred: {label_names[pred_labels[correct_indices[0]]]}')
plt.axis('off')

# true_not_cat_pred_cat 第一張圖
plt.subplot(1, 2, 2)
plt.imshow(test_image2[wrong_indices[0]], cmap=plt.get_cmap('gray'))
plt.title(f'True: {label_names[test_label2[wrong_indices[0]]]}\nPred: {label_names[pred_labels[wrong_indices[0]]]}')
plt.axis('off')

plt.tight_layout()
plt.show()


# The above plots are examples of correctly classified and miss-classified images.\
# It show a true label and predicted label of cat and another example where the true label is not cat but the predicted label is cat.\
# I think the background of the truck picture has a pattern looks like cat ears, so it is miss-classified.

# # 2-3 Following 2-2, observe the feature maps from different convolutional layers and describe how a feature map changes with increasing depth.

# In[358]:


from tensorflow.keras.models import Model


# Plot the feature maps
plt.figure(figsize=(2, 2))

# Plot the original images for both cases

plt.imshow(test_image2[correct_indices[0]], cmap='gray')
plt.title(f'True: {test_label2[correct_indices[0]]}\nPred: {pred_labels[correct_indices[0]]}')
plt.axis('off')


# Choose an input image from the test set
input_image_1 = test_image2[correct_indices[0]].reshape(1,32,32,3)
input_image_2 = test_image2[wrong_indices[0]].reshape(1,32,32,3)

# Get feature maps for the last convolutional layer in each block
ixs = [0, 2]
outputs = [model2_1.layers[i].output for i in ixs]
model = tf.keras.Model(inputs=model2_1.input, outputs=outputs)
feature_maps_1 = model.predict(input_image_1)
feature_maps_2 = model.predict(input_image_2)
layer_names_2 = [layer.name for layer in model2_1.layers[:2] if 'conv2d' in layer.name ]

j = 1  
for layer_name, fmap in zip(layer_names_2, feature_maps_1):
        plt.figure(figsize=(8, 5))  
        for i in range(32):  
            plt.subplot(4, 8, i + 1)
            plt.imshow(fmap[0, :, :, i], cmap='gray')
            plt.axis('off')
            plt.title(f' {i+1}')
        plt.suptitle(f'Feature Maps - conv_{j}')
        j += 1
        plt.show()


# From the above plot, we can find that:
# 
# 1.   Feature maps of convolution layer1 capture some detail in the image, such as edges, corners, and textures.
#      we can see the cat body Contour at the 22th plot.
# 

# In[359]:


# Plot the feature maps
plt.figure(figsize=(2, 2))

# Plot the original images for both cases

plt.imshow(test_image2[wrong_indices[0]], cmap='gray')
plt.title(f'True: {test_label2[wrong_indices[0]]}\nPred: {pred_labels[wrong_indices[0]]}')
plt.axis('off')

j = 1  
for layer_name, fmap in zip(layer_names_2, feature_maps_2):
        plt.figure(figsize=(8, 5))  
        for i in range(32):  
            plt.subplot(4, 8, i + 1)
            plt.imshow(fmap[0, :, :, i], cmap='gray')
            plt.axis('off')
            plt.title(f' {i+1}')
        plt.suptitle(f'Feature Maps - conv_{j}')
        j += 1
        plt.show()


# From the above plot, we can find that:
# 
# 1.   Feature maps of convolution layer1 capture some detail in the image, such as edges, corners, and textures.\
#      We can see that the 22th, 30th plots look like a cat body Contour, so maybe that's the reason of being mis-classified.
# 

# # 2-4 Following 2-1, please add L2 regularization to the CNN implemented in 2-1 and discuss its effect.
# 

# We set $α=0.05$ of L2 regularization.

# * Epochs = 200
# * batch size = 500
# * strides=(3,3),filter=(5,5)
# * kernel_regularizer=0.05
# * activation='relu','softmax'(output)
# * Optimizer = Adam

# In[326]:


model2_L2 = models.Sequential()

#The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
model2_L2.add(layers.Conv2D(64, (2, 2), strides=(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.05), input_shape=(32, 32, 3)))
model2_L2.add(layers.MaxPool2D(((2, 2))))
model2_L2.add(layers.Conv2D(64, (2, 2), strides=(3,3),padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.05)))
model2_L2.add(layers.MaxPool2D(((2, 2))))

model2_L2.add(layers.Flatten())
model2_L2.add(layers.Dense(384, activation='relu' ,kernel_regularizer=regularizers.l2(0.05)))
model2_L2.add(layers.Dropout(0.2))
model2_L2.add(layers.Dense(192, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
model2_L2.add(layers.Dropout(0.2))
model2_L2.add(layers.Dense(10, activation='softmax'))
model2_L2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2_L2.summary()


# In[327]:


from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        print(f'\nTest accuracy after epoch {epoch + 1}: {test_acc:.4f}')
        self.test_accuracy.append(test_acc)


test_callback2_L2 = TestAccuracyCallback(test_data=(test_image2, test_label2))


# In[328]:


history2_L2 = model2_L2.fit(train_image2, train_label2, batch_size=500, epochs=200, validation_split=5000/50000, callbacks=[test_callback2_L2])


# In[329]:


import seaborn as sns

# Plotting training, validation, and testing accuracy
plt.figure(figsize=(9.15,7/3))
plt.subplot(1, 2, 1)
plt.plot(history2_L2.history['accuracy'],color='blue', label='Training Accuracy')
plt.plot(history2_L2.history['val_accuracy'],color='red', label='Validation Accuracy')
plt.plot(test_callback2_L2.test_accuracy,color='green', label='Testing Accuracy')
plt.title('Training, Validation, and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting learning curve
plt.subplot(1, 2, 2)
plt.plot(history2_L2.history['loss'],color='blue', label='Cross entropy')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

## PLOT histograms
all_weights2_L2 = []

for layer in model2_L2.layers:
     # check if weight exists
    if layer.get_weights():
        weights, _ = layer.get_weights()  # ignore bias
        all_weights2_L2.append(weights.flatten())

# 绘制直方图的函数
def plot_weights_histogram(weights_list):
    num_layers = len(weights_list)

    # Calculate the number of rows and columns based on the number of histograms
    num_rows = (num_layers - 1) // 2 + 1
    num_cols = min(num_layers, 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 7))  # Dynamic grid size

    for i in range(num_layers):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index

        axes[row, col].hist(weights_list[i], bins=100, linewidth=1.2)

        if i == 2:
            axes[row, col].set_title(f'Histogram of Dense 1')
        elif i == 3:
            axes[row, col].set_title(f'Histogram of Dense 2')
        elif i == 4:
            axes[row, col].set_title(f'Histogram of Output')
        else:
            axes[row, col].set_title(f'Histogram of Layer {i + 1}')

        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Number')

    for i in range(num_layers, num_rows * num_cols):
        row = i // 2  # Integer division to determine the row index
        col = i % 2   # Modulus operation to determine the column index
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

plot_weights_histogram(all_weights2_L2)

plt.show()


# We can find that:
# 
# 1.  The accuracy rate of train, test, validation are closed.
# 2.  The regularization term encourages the network to learn smaller weight values.
# 3.  The histograms of weights appear to be concentrated around a narrow range, and the value appear to 0.
