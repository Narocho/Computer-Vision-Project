import os
import shutil
import random

# Define the directory for frames
frames_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\frames"

# Define the output directories for train, validation, and test
train_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\train"
val_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\val"
test_dir = r"C:\Users\ADMIN\Documents\New folder\Ms DSE\COS 573\HWK\project\test"

# Create the output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Function to split frames
def split_frames(frames_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Get a list of all frame files in the frames directory
    frame_files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]

    # Shuffle the frame files to ensure random selection
    random.shuffle(frame_files)

    # Calculate split sizes
    total_frames = len(frame_files)
    train_size = int(total_frames * train_ratio)
    val_size = int(total_frames * val_ratio)
    test_size = total_frames - train_size - val_size  # The rest goes to test

    # Split the frame files into train, validation, and test
    train_files = frame_files[:train_size]
    val_files = frame_files[train_size:train_size + val_size]
    test_files = frame_files[train_size + val_size:]

    # Copy the files to their respective directories
    for file in train_files:
        shutil.copy(os.path.join(frames_dir, file), os.path.join(train_dir, file))

    for file in val_files:
        shutil.copy(os.path.join(frames_dir, file), os.path.join(val_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(frames_dir, file), os.path.join(test_dir, file))

    print(f"Total frames: {total_frames}")
    print(f"Training frames: {len(train_files)}")
    print(f"Validation frames: {len(val_files)}")
    print(f"Test frames: {len(test_files)}")


# Call the function to split frames
split_frames(frames_dir, train_dir, val_dir, test_dir)
