import cv2
import numpy as np
import glob
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageDraw as ImageDraw
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import ImageFont



# Initialize the DeepSort tracker with a confidence threshold
tracker = DeepSort(max_age=50)

# Map class IDs to class names
class_names = {
    1: "Person",
    2: "Cyclist",
    3: "Car",
    4: "Unknown",
    5: "Unknown",
    6: "Unknown",
    7: "Unknown",
    8: "Unknown"
}

def save_image_with_boxes_and_ids(image_path, predictions, save_path, confidence_threshold=0.9):
    """
    Draws bounding boxes on the image with class names and saves the new image, filtering predictions by confidence threshold.
    
    Parameters:
    - image_path: str, path to the input image
    - predictions: list of predictions containing boxes, scores, and labels
    - save_path: str, path where to save the image with bounding boxes and class names
    - confidence_threshold: float, minimum confidence score for a prediction to be drawn
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    frame = np.array(image)
        
    font = ImageFont.truetype("perception/fonts/Roboto-Bold.ttf", 17)  # Specify font and size

    # Prepare the list of bounding boxes to be passed to DeepSort
    bbs = []
    for ([left, top, w, h], score, label) in predictions:
        if score >= confidence_threshold:  # Apply the threshold here
            # Append the bounding box and other info (left, top, w, h, confidence, detection_class)
            bbs.append(([left, top, w, h], score, label))

    # Update DeepSort with the new bounding boxes for tracking
    tracks = tracker.update_tracks(bbs, frame=frame)

    # Loop through the tracked objects and draw their bounding boxes and class names
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in [left, top, right, bottom] format
        class_name = class_names.get(track.det_class, "Unknown")  # Get class name
        
        # Ensure the bounding box coordinates are valid
        l, t, r, b = ltrb
        if l > r or t > b:
            print(f"Skipping invalid bounding box: {ltrb}")
            continue
        # Draw the bounding box
        draw.rectangle([ltrb[0], ltrb[1], ltrb[2], ltrb[3]], outline="red", width=3)

        # Draw the class name
        draw.text((ltrb[0]- 10, ltrb[1] - 17), f"{class_name}", fill="red", font=font)
    # Save the image with bounding boxes
    image.save(save_path)
    print(f"Saved image with boxes to {save_path}")

# Camera parameters (intrinsics and extrinsics)
# Replace with actual calibration data. [fx, 0, cx], [0, fy, cy], [0, 0, 1]
K_left = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02], 
                   [0.000000e+00, 9.522352e+02, 2.386081e+02], 
                   [0.000000e+00, 0.000000e+00, 1.000000e+00]])
K_right = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02], 
                    [0.000000e+00, 8.970639e+02, 2.377447e+02], 
                    [0.000000e+00, 0.000000e+00, 1.000000e+00]])
dist_coeff_left = np.zeros((5, 1))  # Rectifyied no distortion; 
dist_coeff_right = np.zeros((5, 1))
R = np.array([[0.99947832, 0.02166116, -0.02395787], # Rotation matrix from left to right camera
              [-0.02162283, 0.99976448, 0.00185789],
              [0.02399247, -0.00133888, 0.99971125]])  
T = np.array([-0.53552388,  0.00666445, -0.01007482])  # Translation vector from left to right camera

# Specify the path to the images, copy the path of images there
#left_image_path = 'perception/34759_final_project_rect/seq_01/image_02/data/*.png'
#right_image_path = 'perception/34759_final_project_rect/seq_01/image_03/data/*.png'
#left_image_path = 'perception/34759_final_project_rect/seq_02/image_02/data/*.png'
#right_image_path = 'perception/34759_final_project_rect/seq_02/image_03/data/*.png'
left_image_path = 'perception/34759_final_project_rect/seq_03/image_02/data/*.png'
right_image_path = 'perception/34759_final_project_rect/seq_03/image_03/data/*.png'
# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))

# Load the pre-trained Faster R-CNN model
model_data = torch.load('perception/models/fasterrcnn_best_epoch_7.pth')
# Initialize a new model with the same architecture
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=8)  # Replace <num_classes>
# Load the state dictionary
model.load_state_dict(model_data)

model.eval()  # Set the model to evaluation mode

# Example of processing and saving images
for i in range(num_images):
    image_path = left_images[i]
    image_tensor = F.to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[0]["boxes"]
    scores = predictions[0]["scores"]
    labels = predictions[0]["labels"]
    
    # Prepare formatted predictions, filtering by confidence threshold
    formatted_predictions = []
    for box, score, label in zip(boxes, scores, labels):
        if score.item() >= 0.9:  # Apply the threshold here
            xmin, ymin, xmax, ymax = box
            left = xmin.item()
            top = ymin.item()
            w = (xmax - xmin).item()
            h = (ymax - ymin).item()
            formatted_predictions.append(([left, top, w, h], score.item(), label.item()))
    
    # Define the path to save the image
    save_path = f"perception/images/frames3/{i}.png"
    
    # Call the function to save the image with bounding boxes
    save_image_with_boxes_and_ids(image_path, formatted_predictions, save_path)

