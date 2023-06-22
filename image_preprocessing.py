import cv2
import imutils
import numpy as np
import os
import string

def process_frame(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a binary mask for the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply a series of morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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
    # Folder path containing the subfolders (A, B, C, ..., Z)
    train_folder_path = "train"

    # Create a new folder to store the processed images
    processed_folder_path = "processed"
    os.makedirs(processed_folder_path, exist_ok=True)

    # Get a list of subfolders (A, B, C, ..., Z)
    subfolders = [folder for folder in string.ascii_uppercase if os.path.isdir(os.path.join(train_folder_path, folder))]

    # Iterate through the subfolders
    for subfolder in subfolders:
        # Create a new folder within the "processed" folder with the same subfolder name
        processed_subfolder_path = os.path.join(processed_folder_path, subfolder)
        os.makedirs(processed_subfolder_path, exist_ok=True)

        # Get the folder path for the current subfolder
        subfolder_path = os.path.join(train_folder_path, subfolder)

        # Get a list of image file names in the current subfolder
        image_files = os.listdir(subfolder_path)

        # Iterate through the image files
        for file_name in image_files:
            # Read the image file
            image_path = os.path.join(subfolder_path, file_name)
            frame = cv2.imread(image_path)

            # Check if the image was read successfully
            if frame is not None:
                # Resize the frame
                frame = imutils.resize(frame, width=700)

                # Process the frame
                processed_frame = process_frame(frame)

                # Save the processed frame in the corresponding subfolder within the "processed" folder
                processed_image_path = os.path.join(processed_subfolder_path, file_name)
                cv2.imwrite(processed_image_path, processed_frame)
                print(f"Processed image saved: {processed_image_path}")

            else:
                print(f"[Warning!] Failed to read image: {file_name}")

    print("All images processed and saved.")

main()