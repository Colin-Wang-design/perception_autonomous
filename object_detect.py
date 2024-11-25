import cv2
import numpy as np
import glob
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import PIL.ImageDraw as ImageDraw

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the path to the images
data_path = "/dtu/blackhole/1b/203515/perception/data"

# Define paths for left and right images
#left_image_path = 'rect/seq_01/image_02/data/*.png'
#right_image_path = 'rect/seq_01/image_03/data/*.png'
# left_image_path = 'rect/seq_02/image_02/data/*.png'
# right_image_path = 'rect/seq_02/image_03/data/*.png'
left_image_path = 'rect/seq_03/image_02/data/*.png'
right_image_path = 'rect/seq_03/image_03/data/*.png'

left_image_path = os.path.join(data_path, left_image_path)
right_image_path = os.path.join(data_path, right_image_path)

# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))

# Load the pre-trained Faster R-CNN model and move it to the GPU
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # Updated to match torchvision 0.13+
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Output directory for saving results
output_dir = 'output/images/seq3/'
output_dir = os.path.join(data_path, output_dir)
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images
for i in range(num_images):
    # Load and process the image
    image = Image.open(left_images[i]).convert("RGB")  # Ensure the image is in RGB format
    image_tensor = F.to_tensor(image).to(device)  # Convert the PIL image to a tensor and move to GPU

    # Make predictions
    with torch.no_grad():
        predictions = model([image_tensor])

    # Convert predictions back to CPU for further processing
    predictions = [{k: v.cpu() for k, v in t.items()} for t in predictions]

    # Create a drawing context on the image
    draw = ImageDraw.Draw(image)

    # Iterate over the predictions and draw bounding boxes
    for idx, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][idx]
        label = predictions[0]['labels'][idx]

        if score > 0.9:  # Confidence threshold
            # Convert the box coordinates from tensor to integer
            xmin, ymin, xmax, ymax = box.int().tolist()

            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            # Optionally, add label and score to the bounding box
            label_text = f"Label {label}, {score:.2f}"
            draw.text((xmin, ymin), label_text, fill="red")

            print(f"Prediction {idx}: Label {label}, Score {score:.2f}")

    # Save the image with bounding boxes
    output_path = os.path.join(output_dir, f"{i}.png")
    image.save(output_path)
    print(f"Saved image {i} with predictions to {output_path}")
    