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
dist_coeff_left = np.zeros((5, 1))  # Rectifyied no distortion; 
dist_coeff_right = np.zeros((5, 1))
R = np.array([[0.99947832, 0.02166116, -0.02395787], # Rotation matrix from left to right camera
              [-0.02162283, 0.99976448, 0.00185789],
              [0.02399247, -0.00133888, 0.99971125]])  
T = np.array([-0.53552388,  0.00666445, -0.01007482])  # Translation vector from left to right camera

# Specify the path to the images, copy the path of images there
#sequn01
left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_02/data/*.png'
right_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_01/image_03/data/*.png'
# left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_02/image_02/data/*.png'
# right_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_02/image_03/data/*.png'
# left_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_03/image_02/data/*.png'
# right_image_path = '/Users/wangdepei/Documents/perception_autonomous/final_project/34759_final_project_rect/seq_03/image_03/data/*.png'

# Read all images with the specified naming convention
left_images = sorted(glob.glob(left_image_path))
right_images = sorted(glob.glob(right_image_path))

# Ensure the number of images in both directories is the same
num_images = min(len(left_images), len(right_images))
if not num_images:
    print("Can not open images")
# print(num_images)

# # Initialize the background subtractor
# get background subtractor

sub_type = 'MOG2' # 'KNN'
if sub_type == "MOG2":
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history = 400,varThreshold=80, detectShadows=True)
    bg_subtractor.setShadowThreshold(0.6)
else:
    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=500, detectShadows=True)
    bg_subtractor.setShadowThreshold(0.20)

# bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
# Define the projection matrices for the left and right cameras
P_left = np.dot(K_left, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Left camera at the origin
P_right = np.dot(K_right, np.hstack((R, T.reshape(3, 1))))  # Right camera with R and T

# Load rectified images in a loop
for i in range(num_images):
    left_img = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
    
    # right_img = cv2.GaussianBlur(right_img, (5, 5), 0)
    # Detect moving object in the left image using background subtraction
    mask = bg_subtractor.apply(right_img)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)  # Threshold to remove shadows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)   # Opening: removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)  # Closing: fills small holes
    mask = cv2.dilate(mask, kernel1, iterations=2)
    
    # Find contours of the detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Filter small contours based on area
        if cv2.contourArea(contour) < 600:
            continue
        
        # Compute bounding box for the detected object
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(right_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Center point in the left image
        point_left = (x + w // 2, y + h // 2)

        # # Corresponding point in the right image (block matching)
        # disparity = cv2.matchTemplate(right_img, left_img[y:y+h, x:x+w], cv2.TM_SQDIFF)
        # min_val, _, min_loc, _ = cv2.minMaxLoc(disparity)
        # point_right = (min_loc[0] + w // 2, y + h // 2)
            
        # # Prepare points for triangulation (must be in homogeneous coordinates)
        # pts_left = np.array([[point_left[0]], [point_left[1]]], dtype=np.float32)
        # pts_right = np.array([[point_right[0]], [point_right[1]]], dtype=np.float32)

        # # Triangulate points to get 3D coordinates
        # points_4D = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
        # points_3D = points_4D[:3] / points_4D[3]  # Convert from homogeneous to 3D

        # # Draw the 3D coordinates
        # print(f"3D Coordinates: {points_3D[0:3].flatten()}")
    # Display the current frame with motion rectangles
    cv2.namedWindow('Mask')        # Create a named window
    cv2.moveWindow('Mask', 40,300)
    cv2.imshow('Mask', mask)

    cv2.imshow('BG subtrac', right_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
