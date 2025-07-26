import os
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define paths
train_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\train"

# Define transforms for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a pre-trained ResNet model for feature extraction
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()  # Set to evaluation mode


# Function to extract features from an image using the CNN
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features


# Function to calculate pupil diameter using the minimum enclosing circle
def detect_pupil_diameter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = radius * 2
        return diameter
    return 0  # No pupil detected


# Function to detect blinking based on pupil diameter
def detect_blinking(curr_pupil_diameter):
    return curr_pupil_diameter < 5  # Threshold for a blink (diameter close to zero)


# Function to detect squinting (based on pupil size changes)
def detect_squinting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter = radius * 2
        # Squinting may correspond to a small but non-zero pupil diameter
        return 5 < diameter < 20  # Adjust thresholds as needed for squinting
    return False


# Function to process frames and extract features
def extract_eye_features_from_frames(directory):
    features = []
    labels = []  # Use participant labels if needed for classification
    pupil_diameters = []  # To store pupil diameters
    squinting_detected = []  # To store squinting status
    blinking_detected = []  # To store blinking status

    for participant in os.listdir(directory):
        participant_dir = os.path.join(directory, participant)
        if os.path.isdir(participant_dir):
            for video_frame in os.listdir(participant_dir):
                frame_path = os.path.join(participant_dir, video_frame)
                if frame_path.endswith(('.jpg', '.png')):
                    eye_img = cv2.imread(frame_path)

                    # Extract CNN features
                    cnn_features = extract_features(frame_path)
                    features.append(cnn_features)

                    # Extract pupil diameter
                    pupil_diameter = detect_pupil_diameter(eye_img)
                    pupil_diameters.append(pupil_diameter)
                    print(f"Pupil Diameter: {pupil_diameter:.4f}")

                    # Check for squinting
                    squinting = detect_squinting(eye_img)
                    if squinting:
                        print(f"Squinting detected in frame: {frame_path}")
                    squinting_detected.append(squinting)

                    # Check for blinking
                    blinking = detect_blinking(pupil_diameter)
                    if blinking:
                        print(f"Blinking detected in frame: {frame_path}")
                    blinking_detected.append(blinking)

                    labels.append(participant)  # Optional label for classification

    smoothed_pupil_diameters = smooth_pupil_diameters(pupil_diameters)
    return features, labels, smoothed_pupil_diameters, squinting_detected, blinking_detected


# Function to smooth pupil diameters (e.g., moving average)
def smooth_pupil_diameters(diameters, window_size=5):
    smoothed = []
    for i in range(len(diameters)):
        if i < window_size:
            smoothed.append(np.mean(diameters[:i+1]))  # Average for the initial frames
        else:
            smoothed.append(np.mean(diameters[i-window_size:i+1]))  # Moving average
    return smoothed


# Extract features for the training set
train_features, train_labels, smoothed_train_pupil_diameters, train_squinting, train_blinking = extract_eye_features_from_frames(train_dir)

# Save the extracted features
torch.save(train_features, 'train_eye_features.pt')
torch.save(train_labels, 'train_labels.pt')
np.save('train_pupil_diameters.npy', smoothed_train_pupil_diameters)  # Save smoothed diameters
np.save('train_squinting.npy', train_squinting)  # Save squinting detection results
np.save('train_blinking.npy', train_blinking)  # Save blinking detection results
