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

def update(X, P, Z, H, R, I):
    y = Z - np.dot(H, X)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))
    x_prime = X + np.dot(K, y)
    p_prime = np.dot((I - np.dot(K, H)), P)
    return (x_prime, p_prime)

# Initialize Kalman filter
kalman_filters = []
def initialize_kalman(x,y):
    # [x, y, dx, dy, ddx, ddy] - position, velocity, and acceleration in x and y directions
    x = np.array([[x], [y], [0], [0], [0], [0]])
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
     # The identity matrix
    I = np.eye(6)
    return x, P, u, F, H, R, I

# Specify the path to the images, copy the path of images there
left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_03/image_02/data/*.png'
right_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_03/data/*.png'
# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))

# Load the pre-trained model
model = YOLO('yolo11s.pt')  # YOLO model yolo11n.pt yolov8n.pt
# Initialize a list to store Kalman filters and associated object IDs
kalman_filters = []
object_ids = []
next_object_id = 0
detection_counts = {}  # Dictionary to count detections for each object ID
distance_threshold = 50 # Tracking threshold for distance
occlusion_frame_threshold = 10  # Number of frames allowed for occlusion
occlusion_counters = {}  # Tracks the number of consecutive occlusion frames per object ID

#for i in range num_images
for i in range(num_images):
    image = Image.open(left_images[i]).convert("RGB")  # Ensure the image is in RGB format
    # Convert the PIL image to a NumPy array for YOLO inference
    image_np = np.array(image)

    # Make predictions
    results = model.predict(image_np)  # Run YOLO inference

    # Extract boxes, scores, and labels from results
    boxes = results[0].boxes.xyxy.numpy() if len(results[0].boxes) > 0 else []  # Bounding boxes
    scores = results[0].boxes.conf.numpy() if len(results[0].boxes) > 0 else [] # Confidence scores
    # labels = results[0].boxes.cls.numpy() if len(results[0].boxes) > 0 else [] # Class labels 
    # Create a drawing context on the image
    draw = ImageDraw.Draw(image)
    # List to store updated filters for this frame
    updated_filters = []
    updated_object_ids = []
    detected_indices = set()
    # Iterate over the predictions and draw bounding boxes
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        if score > 0.5:  # Confidence threshold
            # Convert the box coordinates from tensor to integer
            xmin, ymin, xmax, ymax = map(int, box)

            # Compute the center of the box
            x_pos, y_pos = xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2

            # Match the detection to an existing Kalman filter or create a new one
            matched = False
            for j, (kf, obj_id) in enumerate(zip(kalman_filters, object_ids)):
                x, P, u, F, H, R, I = kf
                x_pred, y_pred = int(x[0][0]), int(x[1][0])
                distance = np.sqrt((x_pos - x_pred) ** 2 + (y_pos - y_pred) ** 2)
                if distance < distance_threshold:
                    # Update Kalman filter with observation
                    Z = np.array([[x_pos], [y_pos]])
                    (x, P) = update(x, P, Z, H, R, I)
                    (x, P) = predict(x, P, F, u)
                    updated_filters.append((x, P, u, F, H, R, I))
                    updated_object_ids.append(obj_id)
                    detection_counts[obj_id] = detection_counts.get(obj_id, 0) + 1
                    occlusion_counters[obj_id] = 0  # Reset occlusion counter
                    matched = True
                    detected_indices.add(j)
                    break

            if not matched:
                # Initialize a new Kalman filter for unmatched detection
                X, P, u, F, H, R, I = initialize_kalman(x_pos, y_pos)
                updated_filters.append((X, P, u, F, H, R, I))
                updated_object_ids.append(next_object_id)
                detection_counts[next_object_id] = 1
                occlusion_counters[next_object_id] = 0
                next_object_id += 1
            
            # Draw the bounding box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            # Draw the predicted position rectangle
            # Draw the predicted position
            x_pred, y_pred = int(updated_filters[-1][0][0][0]), int(updated_filters[-1][0][1][0])
            predicted_box = [x_pred - (xmax - xmin) // 2, y_pred - (ymax - ymin) // 2,
                             x_pred + (xmax - xmin) // 2, y_pred + (ymax - ymin) // 2]
            draw.rectangle(predicted_box, outline="blue", width=2)


    # Predict for unmatched Kalman filters
    for j, (kf, obj_id) in enumerate(zip(kalman_filters, object_ids)):
        if j not in detected_indices:
            x, P, u, F, H, R, I = kf
            occlusion_counters[obj_id] = occlusion_counters.get(obj_id, 0) + 1
            if occlusion_counters[obj_id] <= occlusion_frame_threshold:  # Only keep objects with more than one detection
                (x, P) = predict(x, P, F, u)
                # Draw the predicted position rectangle
                x_pred, y_pred = int(x[0][0]), int(x[1][0])
                predicted_box = [x_pred - (xmax - xmin) // 2, y_pred - (ymax - ymin) // 2,
                                x_pred + (xmax - xmin) // 2, y_pred + (ymax - ymin) // 2]
                # draw.rectangle(predicted_box, outline="green", width=2)
        
                updated_filters.append((x, P, u, F, H, R, I))
                updated_object_ids.append(obj_id)
            elif obj_id in occlusion_counters:
                # Remove the object if occlusion exceeds threshold
                del occlusion_counters[obj_id]

    # Update Kalman filters and object IDs for the next frame
    kalman_filters = updated_filters
    object_ids = updated_object_ids

    # Convert the PIL image back to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Display the image using OpenCV
    cv2.imshow("Detection with Kalman Filter", open_cv_image)
    # Wait for a key press to proceed to the next image
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit early
        break

# Destroy all OpenCV windows after the loop
cv2.destroyAllWindows()

    