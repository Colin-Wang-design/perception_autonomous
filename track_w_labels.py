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

# Specify the path to the images
left_image_path = '/Users/maxbrazhnyy/GitHub/34759_pfas/Project/34759_final_project_rect/seq_01/image_02/data/*.png'
right_image_path = '/Users/maxbrazhnyy/GitHub/34759_pfas/Project/34759_final_project_rect/seq_01/image_03/data/*.png'

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
            bbox = list(map(float, parts[6:10]))  # Extract bbox coordinates
            if frame not in labels:
                labels[frame] = []
            labels[frame].append(bbox)
    return labels

# Load labels
labels_file = '/Users/maxbrazhnyy/GitHub/34759_pfas/Project/34759_final_project_rect/seq_01/labels.txt'  # Fill in the path to the labels file
labels = parse_labels(labels_file)

# Load rectified images in a loop
for i in range(num_images):
    # Load the stereo pair (rectified images)
    left_img = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
    
    # Draw bounding boxes from labels
    if i in labels:
        for bbox in labels[i]:
            left, top, right, bottom = map(int, bbox)
            cv2.rectangle(left_img, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Center point in the left image
            point_left = (left + (right - left) // 2, top + (bottom - top) // 2)

            # Corresponding point in the right image (block matching)
            disparity = cv2.matchTemplate(right_img, left_img[top:bottom, left:right], cv2.TM_SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(disparity)
            point_right = (min_loc[0] + (right - left) // 2, top + (bottom - top) // 2)
                
            # Prepare points for triangulation (must be in homogeneous coordinates)
            pts_left = np.array([[point_left[0]], [point_left[1]]], dtype=np.float32)
            pts_right = np.array([[point_right[0]], [point_right[1]]], dtype=np.float32)

            # Triangulate points to get 3D coordinates
            points_4D = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
            points_3D = points_4D[:3] / points_4D[3]  # Convert from homogeneous to 3D

            # Draw the 3D coordinates
            print(f"3D Coordinates: {points_3D[0:3].flatten()}")

    # Display images
    cv2.imshow('Left Image', left_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
