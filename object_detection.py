import cv2
import numpy as np
import glob
from ultralytics import YOLO
from PIL import Image
import PIL.ImageDraw as ImageDraw

# Kalman filter functions
def predict(X, P, F, u):
    x_prime = np.dot(F, X) + u
    p_prime = np.dot(np.dot(F, P), np.transpose(F))
    return (x_prime, p_prime)

def update(X, P, Z, H, R):
    y = Z - np.dot(H, X)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))
    x_prime = X + np.dot(K, y)
    p_prime = np.dot((np.eye(len(P)) - np.dot(K, H)), P)
    return (x_prime, p_prime)

# Initialize Kalman filter
x = np.array([[0], [0], [0], [0], [0], [0]])
P = np.eye(6) * 10000
u = np.array([[0], [0], [0], [0], [0], [0]])
F = np.array([[1, 0, 1, 0, 0.5, 0],
              [0, 1, 0, 1, 0, 0.5],
              [0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])
R = np.eye(2) * 1

# Specify the path to the images, copy the path of images there
left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_02/data/*.png'
right_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_03/data/*.png'
# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))

# Load the pre-trained model
model = YOLO('yolov8n.pt')  # YOLO model   yolo11n.pt

#for i in range num_images
# image_path = "perception/34759_final_project_rect/seq_01/image_02/data/000000.png"

for i in range(num_images):
    image = Image.open(left_images[i]).convert("RGB")  # Ensure the image is in RGB format
    # Convert the PIL image to a NumPy array for YOLO inference
    image_np = np.array(image)

    # Make predictions
    results = model.predict(image_np)  # Run YOLO inference

    # Extract boxes, scores, and labels from results
    boxes = results[0].boxes.xyxy.numpy()  # Bounding boxes
    scores = results[0].boxes.conf.numpy()  # Confidence scores
    labels = results[0].boxes.cls.numpy()  # Class labels 
    # Create a drawing context on the image
    draw = ImageDraw.Draw(image)

    # Iterate over the predictions and draw bounding boxes
    detected = False
    for box, score, label in zip(boxes, scores, labels):
        
        if score > 0.4:  # Confidence threshold
            detected = True
            # Convert the box coordinates from tensor to integer
            xmin, ymin, xmax, ymax = map(int, box)

            # Compute the center of the box
            x_pos, y_pos = (xmin + xmax) // 2, (ymin + ymax) // 2

            # Update Kalman filter with observation
            Z = np.array([[x_pos], [y_pos]])
            (x, P) = update(x, P, Z, H, R)
            (x, P) = predict(x, P, F, u)

            # Predicted position rectangle
            x_pred, y_pred = int(x[0][0]), int(x[1][0])
            pred_rect_size = 10  # Size of the rectangle

            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            # Optionally, add label and score to the bounding box
            # label_text = f"Label {label}, {score:.2f}"
            # draw.text((xmin, ymin), label_text, fill="red")

            # Draw predicted position rectangle
            # cv2.rectangle(image_np, (x_pred - pred_rect_size, y_pred - pred_rect_size),
            #               (x_pred + pred_rect_size, y_pred + pred_rect_size), (255, 0, 0), 3)
            draw.rectangle([x_pred - pred_rect_size, y_pred - pred_rect_size,
                            x_pred + pred_rect_size, y_pred + pred_rect_size],
                           outline="blue", width=2)

    # If no detection, predict without updating Kalman filter
    if not detected:
        (x, P) = predict(x, P, F, u)
        x_pred, y_pred = int(x[0][0]), int(x[1][0])
        pred_rect_size = 10  # Size of the rectangle
        # cv2.rectangle(image_np, (x_pred - pred_rect_size, y_pred - pred_rect_size),
        #               (x_pred + pred_rect_size, y_pred + pred_rect_size), (255, 0, 0), 3)
        draw.rectangle([x_pred - pred_rect_size, y_pred - pred_rect_size,
                        x_pred + pred_rect_size, y_pred + pred_rect_size],
                       outline="blue", width=2)


    # Convert the PIL image back to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Display the image using OpenCV
    cv2.imshow("Detection with Kalman Filter", open_cv_image)
    # Save the image with bounding boxes
    # output_path = f"/zhome/ec/c/204596/perception/images/seq3/{i}.png"
    # image.save(output_path)
    # Wait for a key press to proceed to the next image
    key = cv2.waitKey(0)
    if key == 27:  # Press 'Esc' to exit early
        break

# Destroy all OpenCV windows after the loop
cv2.destroyAllWindows()

    