import os
import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle

# Path to the "train" folder
train_folder_path = "processed"

# Lists to store training and test images, labels, and output vectors
train_images = []
train_labels = []
train_output_vectors = []

test_images = []
test_labels = []
test_output_vectors = []

# Percentage of images for training set (80%)
train_percentage = 0.8

# Iterate over the subfolders (A, B, C, ..., Z)
for class_index, folder_name in enumerate(sorted(os.listdir(train_folder_path))):
    folder_path = os.path.join(train_folder_path, folder_name)
    if os.path.isdir(folder_path):
        # Get the list of image files in the subfolder
        images = os.listdir(folder_path)
        num_images = len(images)
        num_train = int(num_images * train_percentage)

        # Shuffle the image list
        np.random.shuffle(images)

        # Split the images into training and test sets
        train_images_class = images[:num_train]
        test_images_class = images[num_train:]

        # Load and resize images for training set
        for image_name in train_images_class:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (100, 89))
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            train_images.append(image_resized.reshape(89, 100, 1))
            train_labels.append(class_index)

        # Load and resize images for test set
        for image_name in test_images_class:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (100, 89))
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            test_images.append(image_resized.reshape(89, 100, 1))
            test_labels.append(class_index)

# Convert labels to output vectors using one-hot encoding for training set
train_output_vectors = np.eye(24)[train_labels]

# Convert labels to output vectors using one-hot encoding for test set
test_output_vectors = np.eye(24)[test_labels]


tf.compat.v1.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,24,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Shuffle Training Data
train_images, train_output_vectors = shuffle(train_images, train_output_vectors, random_state=0)

# Train model
model.fit(train_images, train_output_vectors, n_epoch=50,
           validation_set = (test_images, test_output_vectors),
           snapshot_epoch=True, show_metric=True, run_id='convnet_coursera')

model.save("TrainedModel/GestureRecogModel.tfl")