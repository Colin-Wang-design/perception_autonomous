import cv2
import numpy as np
import glob
from ultralytics import YOLO

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
left_image_path = '/Users/wangdepei/Documents/courses/perception_autonomous/final_project/34759_final_project_rect/seq_02/image_02/data/*.png'
right_image_path = '/Users/wangdepei/Documents/courses/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_03/data/*.png'
# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))
if not left_images:
    print("Can not opend imgaes!")
# Ensure the number of images in both directories is the same
# num_images = min(len(left_images), len(right_images))

# Load the pre-trained model
model = YOLO('yolo11s.pt')  # YOLO model yolo11n.pt yolov8n.pt
# Get the class names from the model
class_names = model.names  # Ensure this retrieves the correct names list

# Initialize a list to store Kalman filters and associated object IDs
kalman_filters = []
object_ids = []
next_object_id = 0
detection_counts = {}  # Dictionary to count detections for each object ID
distance_threshold = 70 # Tracking threshold for distance
occlusion_frame_threshold = 12  # Number of frames allowed for occlusion
occlusion_counters = {}  # Tracks the number of consecutive occlusion frames per object ID

#for i in range num_images
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
    
    # Track matched indices
    matched_indices = set()
    detection_used = [False] * len(boxes)
    # List to store updated filters for this frame
    updated_filters = []
    updated_object_ids = []
    detected_indices = set()
    # Match each detection to the closest Kalman filter
    for i, (x, P, u, F, H, R, I) in enumerate(kalman_filters):
        x_pred, y_pred = int(x[0][0]), int(x[1][0])
        best_distance = float('inf')
        best_idx = -1

        # Find the closest detection for this Kalman filter
        for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if detection_used[j] or score <= 0.5 or label >= 4.0:
                continue

            # Compute the center of the detection
            xmin, ymin, xmax, ymax = map(int, box)
            x_pos, y_pos = xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2

            # Compute the distance
            distance = np.sqrt((x_pos - x_pred) ** 2 + (y_pos - y_pred) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_idx = j

        # If a match is found within the distance threshold, update Kalman filter
        if best_idx != -1 and best_distance < distance_threshold:
            xmin, ymin, xmax, ymax = map(int, boxes[best_idx])
            x_pos, y_pos = xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2

            # Update Kalman filter
            Z = np.array([[x_pos], [y_pos]])
            (x, P) = update(x, P, Z, H, R, I)
            (x, P) = predict(x, P, F, u)

            # Mark detection as used
            detection_used[best_idx] = True
            # matched_indices.add(i)
            occlusion_counters[object_ids[i]] = 0  # Reset occlusion counter
        
        else:
            # Predict the Kalman filter state without updating
            (x, P) = predict(x, P, F, u)
            occlusion_counters[object_ids[i]] = occlusion_counters.get(object_ids[i], 0) + 1
        
        updated_filters.append((x, P, u, F, H, R, I))
        updated_object_ids.append(object_ids[i])

    # Create new Kalman filters for unmatched detections        
    for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if detection_used[j] or score <= 0.5 or label >= 4.0:
            continue

        # Compute the center of the detection
        xmin, ymin, xmax, ymax = map(int, box)
        x_pos, y_pos = xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2

        # Initialize a new Kalman filter
        X, P, u, F, H, R, I = initialize_kalman(x_pos, y_pos)
        updated_filters.append((X, P, u, F, H, R, I))
        updated_object_ids.append(next_object_id)

        detection_counts[next_object_id] = 1
        occlusion_counters[next_object_id] = 0
        next_object_id += 1

    # Remove filters that have been stop for too long
    final_filters = []
    final_object_ids = []
    min_position_change = 1  # Minimum position change threshold to keep the filter

    for (kf, obj_id) in zip(updated_filters, updated_object_ids):
        x_pred, y_pred = int(kf[0][0][0]), int(kf[0][1][0])  # Predicted position

        # Retrieve previous position if available
        if f"prev_pos_{obj_id}" in globals():
            prev_x, prev_y = globals()[f"prev_pos_{obj_id}"]
            position_change = np.sqrt((x_pred - prev_x) ** 2 + (y_pred - prev_y) ** 2)
            # print("position_change:",obj_id,position_change)

            # Only keep the filter if the position change is above the threshold
            if position_change > min_position_change:
                final_filters.append(kf)
                final_object_ids.append(obj_id)
                globals()[f"prev_pos_{obj_id}"] = (x_pred, y_pred)  # Update position
        else:
            # First frame for this object, keep the filter
            final_filters.append(kf)
            final_object_ids.append(obj_id)
            globals()[f"prev_pos_{obj_id}"] = (x_pred, y_pred)  # Initialize position
     
     
    # Update Kalman filters and object IDs for the next frame
    kalman_filters = final_filters
    object_ids = final_object_ids

    # Visualize the updated Kalman filters and detections
    for (x, P, _, _, _, _, _), obj_id in zip(kalman_filters, object_ids):
        x_pred, y_pred = int(x[0][0]), int(x[1][0])

        # Draw predicted bounding boxes
        cv2.circle(image, (x_pred, y_pred), 5, (255, 0, 0), -1)  # Blue circle for prediction
        cv2.putText(image, f"ID: {obj_id}", (x_pred + 10, y_pred - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image using OpenCV
    cv2.imshow("Detection with Kalman Filter", image)
    # Wait for a key press to proceed to the next image
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit early
        break

# Destroy all OpenCV windows after the loop
cv2.destroyAllWindows()

    