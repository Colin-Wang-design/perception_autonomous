import cv2
import numpy as np
from ultralytics import YOLO
import glob

# Specify the path to the images
left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_03/image_02/data/*.png'

# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))

# Load the pre-trained YOLO model
model = YOLO('yolo11s.pt')  # YOLO model yolo11n.pt or yolov8n.pt

# Get the class names from the model
class_names = model.names  # Ensure this retrieves the correct names list

# Loop through each image
for img_path in left_images:
    image = cv2.imread(img_path)  # Read the image using OpenCV

    # Convert the image from BGR to RGB for YOLO inference
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make predictions
    results = model.predict(image_rgb)  # Run YOLO inference

    # Extract boxes, scores, and labels from results
    boxes = results[0].boxes.xyxy.numpy() if len(results[0].boxes) > 0 else []  # Bounding boxes
    scores = results[0].boxes.conf.numpy() if len(results[0].boxes) > 0 else [] # Confidence scores
    labels = results[0].boxes.cls.numpy() if len(results[0].boxes) > 0 else [] # Class labels 
    
    # Iterate over the predictions and draw bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5 and label < 4.0:  # Confidence threshold and only car, person and bikes
            # Convert the box coordinates from tensor to integer
            xmin, ymin, xmax, ymax = map(int, box)
            # Get the class name for the label
            class_name =  class_names[int(label)]

            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red box

            # Create label text with the class name and confidence score
            label_text = f"{class_name} ({score:.2f})"

            # Calculate the position for the text
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width, text_height = text_size
            label_bg_xmax = xmin + text_width
            label_bg_ymin = ymin - text_height - 4
            label_bg_ymax = ymin

            # Draw a filled rectangle for the text background
            cv2.rectangle(image, (xmin, label_bg_ymin), (label_bg_xmax, label_bg_ymax), (0, 0, 255), -1)  # Red background

            # Put the text on the image
            cv2.putText(image, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

     # Display the image using OpenCV
    cv2.imshow("YOLO Detection", image)

    # Wait for a key press to proceed to the next image
    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit early
        break

# Destroy all OpenCV windows after the loop
cv2.destroyAllWindows()
