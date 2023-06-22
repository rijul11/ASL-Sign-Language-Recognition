import cv2
import imutils
import numpy as np
import os
import string
import tensorflow as tf
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn


def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    img = img.resize((basewidth, 89), Image.ANTIALIAS)  # Resize to (100, 89)
    img.save(imageName)

def process_frame(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Add weight to the mask to enhance the skin regions
    skinMask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)

    # Apply median blur to the skin mask
    skinMask = cv2.medianBlur(skinMask, 5)

    # Apply bitwise and operation between the original frame and the skin mask
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    # Convert the resulting image to grayscale
    skinGray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to the skin image
    edges = cv2.Canny(skinGray, 60, 60)

    return edges


def main():
    # Load the trained model
    model_path = "TrainedModel/GestureRecogModel.tfl"
    model = tflearn.DNN(load_trained_model())
    model.load(model_path)

    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if ret:
            # Resize the frame
            frame = imutils.resize(frame, width=700)

            # Process the frame
            processed_frame = process_frame(frame)

            # Display the processed frame
            cv2.imshow("Processed Frame", processed_frame)

            # Perform prediction on the processed frame
            prediction = predict_gesture(processed_frame, model)

            # Display the predicted gesture on the frame
            cv2.putText(frame, "Prediction: " + prediction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


def load_trained_model():
    # Path to the trained model file

    # Define the CNN architecture
    convnet = input_data(shape=[None, 89, 100, 1], name='input')
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1000, activation='relu')
    convnet = dropout(convnet, 0.75)

    convnet = fully_connected(convnet, 24, activation='softmax')

    convnet = regression(convnet, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='regression')

    return convnet


def predict_gesture(processed_frame, model):
    # Resize the processed frame to match the input size of the model
    processed_frame = cv2.resize(processed_frame, (100, 89))

    # Reshape the frame to match the model's input shape
    processed_frame = processed_frame.reshape([-1, 89, 100, 1])

    # Perform the prediction using the model
    prediction = model.predict(processed_frame)

    # Get the predicted gesture label
    label = np.argmax(prediction)

    # Map the label to the corresponding gesture
    gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    predicted_gesture = gestures[label]

    return predicted_gesture


# def predict_gesture(image, model):
#     # Preprocess the image
#     # processed_image = cv2.resize(image, (100, 89))
#     # processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
#
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_image = cv2.resize(gray_image, (100, 89))  # Resize to (100, 89)
#     prediction = model.predict([gray_image.reshape(89, 100, 1)])
#
#     # Perform prediction
#     # prediction = model.predict(processed_image.reshape(89, 100, 1))
#     print(len(prediction))
#     # Get the predicted gesture label
#     predicted_label = np.argmax(prediction)
#     gesture_label = get_gesture_label(predicted_label)
#
#     return gesture_label
#
#
# def get_gesture_label(label_index):
#     # List of gesture labels excluding 'J' and 'Z'
#     gesture_labels = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in ['J', 'Z']]
#
#     return gesture_labels[label_index]


if __name__ == "__main__":
    main()