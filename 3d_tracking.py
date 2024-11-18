import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt

# Replace with actual calibration data. [fx, 0, cx], [0, fy, cy], [0, 0, 1]
K_left = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02], 
                   [0.000000e+00, 9.522352e+02, 2.386081e+02], 
                   [0.000000e+00, 0.000000e+00, 1.000000e+00]])
K_right = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02], 
                    [0.000000e+00, 8.970639e+02, 2.377447e+02], 
                    [0.000000e+00, 0.000000e+00, 1.000000e+00]])

R = np.array([[0.99947832, 0.02166116, -0.02395787], # Rotation matrix from left to right camera
              [-0.02162283, 0.99976448, 0.00185789],
              [0.02399247, -0.00133888, 0.99971125]])  
T = np.array([-0.53552388,  0.00666445, -0.01007482])  # Translation vector from left to right camera

# Sequence directory
sequence_path = '/Users/maxbrazhnyy/GitHub/34759_pfas/Project/34759_final_project_rect/seq_01'

# Specify the path to the images
left_image_path = sequence_path + '/image_02/data/*.png'
right_image_path = sequence_path + '/image_03/data/*.png'

# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))

# Define the projection matrices for the left and right cameras
P_left = np.dot(K_left, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Left camera at the origin
P_right = np.dot(K_right, np.hstack((R, T.reshape(3, 1))))  # Right camera with R and T

# Function to parse the labels file
def parse_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            frame = int(parts[0])
            track_id = int(parts[1])
            object_type = parts[2]
            truncated = int(parts[3])
            occluded = int(parts[4])
            alpha = float(parts[5])
            bbox = list(map(float, parts[6:10]))  # Extract bbox coordinates
            dimensions = list(map(float, parts[10:13]))  # Extract dimensions (height, width, length)
            location = list(map(float, parts[13:16]))  # Extract location (x, y, z)
            rotation_y = float(parts[16])
            
            if frame not in labels:
                labels[frame] = []
            labels[frame].append({
                'track_id': track_id,
                'object_type': object_type,
                'truncated': truncated,
                'occluded': occluded,
                'alpha': alpha,
                'bbox': bbox,
                'dimensions': dimensions,
                'location': location,
                'rotation_y': rotation_y
            })
    return labels

# Kalman filter initialization
def initialize_kalman(initial_point):
    # The initial state (6x1).
    # [x, y, dx, dy, ddx, ddy] - position, velocity, and acceleration in x and y directions
    x = np.array([[initial_point[0]], [initial_point[1]], [0], [0], [0], [0]])

    # The initial uncertainty (6x6).
    P = np.eye(6) * 1000

    # The external motion (6x1).
    u = np.array([[0], [0], [0], [0], [0], [0]])

    # The transition matrix (6x6)
    F = np.array([[1, 0, 1, 0, 0.5, 0],
                  [0, 1, 0, 1, 0, 0.5],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    # The observation matrix (2x6)
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0]])

    # The measurement uncertainty.
    R = np.eye(2) * 1

    # The identity matrix
    I = np.eye(6)

    return x, P, u, F, H, R, I

# Kalman filter predict function
def predict(X, P, F, u):
    x_prime = np.dot(F, X) + u
    p_prime = np.dot(np.dot(F, P), np.transpose(F))
    return x_prime, p_prime

# Kalman filter update function
def update(X, P, Z, H, R):
    y = Z - np.dot(H, X)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))
    x_prime = X + np.dot(K, y)
    p_prime = np.dot((I - np.dot(K, H)), P)
    return x_prime, p_prime

# Initialize Kalman filter for each label
kalman_filters = {}

# Load labels
labels_file = sequence_path + '/labels.txt'  # Fill in the path to the labels file
labels = parse_labels(labels_file)

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set fixed limits for the plot
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-1, 19])

# Plot the camera positions with larger markers
sensor_size = np.array([0.036, 0.024])
intrinsic_matrix = np.array([
    [0.05, 0, sensor_size[0] / 2.0],
    [0, 0.05, sensor_size[1] / 2.0],
    [0, 0, 1]
])
virtual_image_distance = 1

# Left camera
cam2world_left = pt.transform_from_pq([0, 0, 0, 0, 0, 0, 1])  # Quaternion for z direction
pc.plot_camera(ax, cam2world=cam2world_left, M=intrinsic_matrix, sensor_size=sensor_size, virtual_image_distance=virtual_image_distance, color='blue')

# Right camera
cam2world_right = pt.transform_from_pq([T[0], T[1], T[2], 0, 0, 0, 1])  # Quaternion for z direction
pc.plot_camera(ax, cam2world=cam2world_right, M=intrinsic_matrix, sensor_size=sensor_size, virtual_image_distance=virtual_image_distance, color='green')

ax.legend()

# Function to update the 3D plot
def update_3d_plot(ax, points_3d, track_ids):
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 19])
    pc.plot_camera(ax, cam2world=cam2world_left, M=intrinsic_matrix, sensor_size=sensor_size, virtual_image_distance=virtual_image_distance, color='blue')
    pc.plot_camera(ax, cam2world=cam2world_right, M=intrinsic_matrix, sensor_size=sensor_size, virtual_image_distance=virtual_image_distance, color='green')
    for point, track_id in zip(points_3d, track_ids):
        if -10 <= point[0] <= 10 and -10 <= point[1] <= 10 and -1 <= point[2] <= 19:
            ax.scatter(point[0], point[1], point[2], c='red', marker='o')
            ax.text(point[0], point[1], point[2], str(track_id), color='red')
    ax.legend()
    plt.draw()
    plt.pause(0.001)

# Variable to control whether to use grayscale or color
use_grayscale = True  # Set to True for grayscale, False for color

# Load rectified images in a loop
for i in range(num_images):
    # Load the stereo pair (rectified images)
    if use_grayscale:
        left_img = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
    else:
        left_img = cv2.imread(left_images[i], cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_images[i], cv2.IMREAD_COLOR)
    
    # Add text to indicate left or right image and frame number
    cv2.putText(left_img, f"Left Image - Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(right_img, f"Right Image - Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw bounding boxes from labels
    points_3d = []
    track_ids = []
    if i in labels:
        for label in labels[i]:
            bbox = label['bbox']
            left, top, right, bottom = map(int, bbox)
            track_id = label['track_id']
            cv2.rectangle(left_img, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.putText(left_img, str(track_id), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Center point in the left image
            point_left = (left + (right - left) // 2, top + (bottom - top) // 2)

            # Initialize Kalman filter if not already initialized
            if label['track_id'] not in kalman_filters:
                kalman_filters[label['track_id']] = initialize_kalman(point_left)

            # Get Kalman filter for the current label
            x, P, u, F, H, R, I = kalman_filters[label['track_id']]

            # Update Kalman filter with the current measurement
            Z = np.array([[point_left[0]], [point_left[1]]])
            x, P = update(x, P, Z, H, R)

            # Predict the next position
            x, P = predict(x, P, F, u)
            predicted_point_left = (int(x[0][0]), int(x[1][0]))

            # Save updated state back to the Kalman filter
            kalman_filters[label['track_id']] = (x, P, u, F, H, R, I)

            # Debugging prints
            print(f"Frame {i}, Track ID {label['track_id']}:")
            print(f"  Current point: {point_left}")
            print(f"  Predicted point: {predicted_point_left}")

            # Calculate the predicted bounding box coordinates
            predicted_left = predicted_point_left[0] - (right - left) // 2
            predicted_top = predicted_point_left[1] - (bottom - top) // 2
            predicted_right = predicted_point_left[0] + (right - left) // 2
            predicted_bottom = predicted_point_left[1] + (bottom - top) // 2

            # Draw the predicted bounding box in red
            cv2.rectangle(left_img, (predicted_left, predicted_top), (predicted_right, predicted_bottom), (0, 0, 255), 2)
            cv2.putText(left_img, str(track_id), (predicted_left, predicted_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Use the 3D location from the labels file
            location = label['location']
            points_3d.append(location)
            track_ids.append(track_id)

            # Draw the 3D coordinates
            print(f"3D Coordinates: {location}")

    # Update the 3D plot
    update_3d_plot(ax, points_3d, track_ids)

    # Stack images vertically
    stacked_img = np.vstack((left_img, right_img))

    # Display stacked images
    cv2.imshow('Stacked Images', stacked_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()