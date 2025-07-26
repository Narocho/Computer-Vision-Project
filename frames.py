import os
import shutil
import random
import cv2

# Define paths
input_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\LPW\LPW"  # Path to LPW dataset
frames_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\frames"  # Path to save frames

# Define the output directories for train, validation, and test
train_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\train"
val_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\val"
test_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\test"

# Create the output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Function to extract frames from a video
def extract_frames_from_video(video_path, participant_name, video_name, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale (optional, depending on your needs)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save the frame to the output directory
        frame_filename = os.path.join(output_dir, f"{participant_name}_{video_name}_frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_gray)

        # Increment the frame count
        frame_count += 1

    cap.release()


# Function to split the dataset into train, validation, and test
def split_data(input_dir, train_dir, val_dir, test_dir, train_size=15, val_size=4, test_size=3):
    # List all participants (folders)
    participants = [p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]

    # Shuffle the participants for random splitting
    random.shuffle(participants)

    # Split the participants into training, validation, and test
    train_participants = participants[:train_size]
    val_participants = participants[train_size:train_size + val_size]
    test_participants = participants[train_size + val_size:]

    # Process videos for each split
    for participant in train_participants:
        participant_dir = os.path.join(input_dir, participant)
        participant_train_dir = os.path.join(train_dir, participant)
        os.makedirs(participant_train_dir, exist_ok=True)

        # Extract frames for each video of this participant
        for video_file in os.listdir(participant_dir):
            if video_file.endswith(('.mp4', '.avi')):  # Process video files
                video_path = os.path.join(participant_dir, video_file)
                extract_frames_from_video(video_path, participant, os.path.splitext(video_file)[0],
                                          participant_train_dir)

    for participant in val_participants:
        participant_dir = os.path.join(input_dir, participant)
        participant_val_dir = os.path.join(val_dir, participant)
        os.makedirs(participant_val_dir, exist_ok=True)

        for video_file in os.listdir(participant_dir):
            if video_file.endswith(('.mp4', '.avi')):  # Process video files
                video_path = os.path.join(participant_dir, video_file)
                extract_frames_from_video(video_path, participant, os.path.splitext(video_file)[0], participant_val_dir)

    for participant in test_participants:
        participant_dir = os.path.join(input_dir, participant)
        participant_test_dir = os.path.join(test_dir, participant)
        os.makedirs(participant_test_dir, exist_ok=True)

        for video_file in os.listdir(participant_dir):
            if video_file.endswith(('.mp4', '.avi')):  # Process video files
                video_path = os.path.join(participant_dir, video_file)
                extract_frames_from_video(video_path, participant, os.path.splitext(video_file)[0],
                                          participant_test_dir)

    print(f"Training participants: {len(train_participants)}")
    print(f"Validation participants: {len(val_participants)}")
    print(f"Test participants: {len(test_participants)}")
    print(f"Frames extracted and saved to respective directories.")


# Call the function to split data and extract frames
split_data(input_dir, train_dir, val_dir, test_dir)
