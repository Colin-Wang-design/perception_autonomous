import cv2
import numpy as np
import glob

# Camera parameters (intrinsics and extrinsics)
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

#sequence directory
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

# Load labels
labels_file = sequence_path + '/labels.txt'  # Fill in the path to the labels file
labels = parse_labels(labels_file)
# Load rectified images in a loop
for i in range(num_images):
    # Load the stereo pair (rectified images)
    left_img = cv2.imread(left_images[i], cv2.IMREAD_COLOR) # for grayscale (and faster processing, use cv2.IMREAD_GRAYSCALE))
    right_img = cv2.imread(right_images[i], cv2.IMREAD_COLOR)
    
   # Add text to indicate left or right image and frame number
    cv2.putText(left_img, f"Left Image - Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(right_img, f"Right Image - Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw bounding boxes from labels
    if i in labels:
        for label in labels[i]:
            bbox = label['bbox']
            left, top, right, bottom = map(int, bbox)
            track_id = label['track_id']
            cv2.rectangle(left_img, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.putText(left_img, str(track_id), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Center point in the left image
            point_left = (left + (right - left) // 2, top + (bottom - top) // 2)

            # Corresponding point in the right image (block matching)
            disparity = cv2.matchTemplate(right_img, left_img[top:bottom, left:right], cv2.TM_SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(disparity)
            point_right = (min_loc[0] + (right - left) // 2, top + (bottom - top) // 2)
            
            # Draw bounding box on the right image
            right_bbox_left = min_loc[0]
            right_bbox_top = top
            right_bbox_right = min_loc[0] + (right - left)
            right_bbox_bottom = bottom
            cv2.rectangle(right_img, (right_bbox_left, right_bbox_top), (right_bbox_right, right_bbox_bottom), (0, 0, 0), 2)
            cv2.putText(right_img, str(track_id), (right_bbox_left, right_bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
            # Prepare points for triangulation (must be in homogeneous coordinates)
            pts_left = np.array([[point_left[0]], [point_left[1]]], dtype=np.float32)
            pts_right = np.array([[point_right[0]], [point_right[1]]], dtype=np.float32)

            # Triangulate points to get 3D coordinates
            points_4D = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
            points_3D = points_4D[:3] / points_4D[3]  # Convert from homogeneous to 3D

            # Draw the 3D coordinates
            print(f"3D Coordinates: {points_3D[0:3].flatten()}")

    # Stack images vertically
    stacked_img = np.vstack((left_img, right_img))

    # Display stacked images
    cv2.imshow('Stacked Images', stacked_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()