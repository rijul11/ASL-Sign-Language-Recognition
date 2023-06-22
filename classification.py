import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Path to the folder containing image data
data_folder_path = "processed"

# Lists to store image data and corresponding labels
images = []
labels = []

# Iterate over the subfolders in the data folder
for class_index, folder_name in enumerate(sorted(os.listdir(data_folder_path))):
    folder_path = os.path.join(data_folder_path, folder_name)
    if os.path.isdir(folder_path):
        # Get the list of image files in the subfolder
        image_files = os.listdir(folder_path)
        # Load and resize images
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (100, 100))
            images.append(image.flatten())
            labels.append(class_index)

# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


def run_GaussianNB():
    model = GaussianNB()
    model.fit(train_images, train_labels)

    # Predict the labels for training and test sets
    train_predictions = model.predict(train_images)
    test_predictions = model.predict(test_images)

    # Calculate accuracy for training and test sets
    train_accuracy = (train_predictions == train_labels).mean()
    test_accuracy = (test_predictions == test_labels).mean()

    print("GaussianNB Train Accuracy:", train_accuracy)
    print("GaussianNB Test Accuracy:", test_accuracy)


def run_KNN():
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_images, train_labels)

    # Predict the labels for training set
    train_predictions = model.predict(train_images)
    train_accuracy = (train_predictions == train_labels).mean()

    # Predict the labels for test set
    test_predictions = model.predict(test_images)
    test_accuracy = (test_predictions == test_labels).mean()

    print("KNN Train Accuracy:", train_accuracy)
    print("KNN Test Accuracy:", test_accuracy)


def run_SVC():
    model = SVC()
    model.fit(train_images, train_labels)

    # Predict the labels for training and test sets
    train_predictions = model.predict(train_images)
    test_predictions = model.predict(test_images)

    # Calculate accuracy for training and test sets
    train_accuracy = (train_predictions == train_labels).mean()
    test_accuracy = (test_predictions == test_labels).mean()

    print("SVC Train Accuracy:", train_accuracy)
    print("SVC Test Accuracy:", test_accuracy)


def run_lr():
    model = LogisticRegression(penalty='l2', solver='saga', C=0.1, max_iter=500)
    model.fit(train_images, train_labels)

    # Predict the labels for training and test sets
    train_predictions = model.predict(train_images)
    test_predictions = model.predict(test_images)

    # Calculate accuracy for training and test sets
    train_accuracy = (train_predictions == train_labels).mean()
    test_accuracy = (test_predictions == test_labels).mean()

    print("LR Train Accuracy:", train_accuracy)
    print("LR Test Accuracy:", test_accuracy)


def run_xgb():
    model = xgb.XGBClassifier(n_estimators=50, max_depth=7, min_child_weight=3)
    model.fit(train_images, train_labels)

    # Predict the labels for training and test sets
    train_predictions = model.predict(train_images)
    test_predictions = model.predict(test_images)

    # Calculate accuracy for training and test sets
    train_accuracy = (train_predictions == train_labels).mean()
    test_accuracy = (test_predictions == test_labels).mean()

    print("XGB Train Accuracy:", train_accuracy)
    print("XGB Test Accuracy:", test_accuracy)


def run_RF():
    # Create and train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=150, min_samples_split=40, min_samples_leaf=15, max_depth=15)
    model.fit(train_images, train_labels)

    # Predict the labels for training and test sets
    train_predictions = model.predict(train_images)
    test_predictions = model.predict(test_images)

    # Calculate accuracy for training and test sets
    train_accuracy = (train_predictions == train_labels).mean()
    test_accuracy = (test_predictions == test_labels).mean()

    print("RF Train Accuracy:", train_accuracy)
    print("RF Test Accuracy:", test_accuracy)


run_GaussianNB()
run_KNN()
run_SVC()
run_xgb()
run_lr()
run_RF()
