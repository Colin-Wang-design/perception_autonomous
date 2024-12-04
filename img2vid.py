import cv2
import os
import glob

# Path to the directory containing images
image_folder = 'perception/images/frames3'
video_name = 'perception/images/frames3.mp4'

# Get all image file paths
images = glob.glob(os.path.join(image_folder, '*.png'))

# Sort the images numerically by the number in the filename
images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Check if there are images in the folder
if not images:
    print("No images found in the specified folder.")
    exit()

# Read the first image to determine the video dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = 10  # Frames per second
out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Write each image to the video
for image in images:
    frame = cv2.imread(image)
    frame = cv2.resize(frame, (width, height))  # Resize to ensure consistency
    out.write(frame)  # Write the frame to the video

# Release the VideoWriter and close all OpenCV windows
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")
