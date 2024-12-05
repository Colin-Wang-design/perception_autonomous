import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytransform3d.transformations as pt
import pytransform3d.camera as pc



# Calibration data directly copied from calib_cam_to_cam.txt
K_left = np.array([
    [9.569475e+02, 0.000000e+00, 6.939767e+02],
    [0.000000e+00, 9.522352e+02, 2.386081e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])

K_right = np.array([
    [9.011007e+02, 0.000000e+00, 6.982947e+02],
    [0.000000e+00, 8.970639e+02, 2.377447e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])

# Extract focal lengths for camera 2
f_x_02 = K_left[0, 0]
f_y_02 = K_left[1, 1]

# Extract focal lengths for camera 3
f_x_03 = K_right[0, 0]
f_y_03 = K_right[1, 1]

print(f"Focal length for camera 2 in x direction: {f_x_02}")
print(f"Focal length for camera 2 in y direction: {f_y_02}")
print(f"Focal length for camera 3 in x direction: {f_x_03}")
print(f"Focal length for camera 3 in y direction: {f_y_03}")
# Calculate the average focal lengths for each camera
avg_focal_length_02 = (f_x_02 + f_y_02) / 2
avg_focal_length_03 = (f_x_03 + f_y_03) / 2

print(f"Average focal length for camera 2: {avg_focal_length_02}")
print(f"Average focal length for camera 3: {avg_focal_length_03}")


R = np.array([
    [0.99947832, 0.02166116, -0.02395787],
    [-0.02162283, 0.99976448, 0.00185789],
    [0.02399247, -0.00133888, 0.99971125]
])



# Update translation vector T to reflect the 0.54 meters separation
T = np.array([-0.54, 0, 0])

# Define projection matrices with updated T
P_left = np.dot(K_left, np.hstack((np.eye(3), np.zeros((3, 1)))))
P_right = np.dot(K_right, np.hstack((R, T.reshape(3, 1))))

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

# Function to parse the labels file
def parse_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            frame = int(parts[0])
            track_id = int(parts[1])
            bbox = tuple(map(float, parts[6:10]))  # Extract bounding box coordinates as floats
            if frame not in labels:
                labels[frame] = []
            labels[frame].append({'track_id': track_id, 'bbox': bbox})
    return labels

# Load labels
labels = parse_labels(sequence_path + '/labels.txt')

def my_triangulate(T, f_02, f_03, pts_02, pts_03):
    # Calculate the disparity
    disparity = pts_02[0] - pts_03[0]
    f = (f_02 + f_03) / 2
    print("f", f)
    # Calculate the depth in meters
    depth = abs(T[0]) * f / disparity
    
    # Calculate the 3D point
    X = np.array([
        depth * (pts_02[0] - K_left[0, 2]) / f_02,
        depth * (pts_02[1] - K_left[1, 2]) / f_02,
        depth
    ])
    
    return X
    
# Toggle to choose triangulation method
use_custom_triangulation = True

# Setup 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set fixed plot limits
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([0, 20])

# Function to plot cameras
def plot_cameras(ax):
    sensor_size = np.array([0.036, 0.024])
    intrinsic_matrix = np.array([
        [0.05, 0, sensor_size[0] / 2.0],
        [0, 0.05, sensor_size[1] / 2.0],
        [0, 0, 1]
    ])
    virtual_image_distance = 1
    
    # Left camera
    cam2world_left = pt.transform_from_pq([0, 0, 0, 0, 0, 0, 1])
    pc.plot_camera(ax, cam2world=cam2world_left, M=intrinsic_matrix, 
                  sensor_size=sensor_size, 
                  virtual_image_distance=virtual_image_distance, 
                  color='blue')
    
    # Right camera
    cam2world_right = pt.transform_from_pq([-0.54, 0, 0, 0, 0, 0, 1])
    pc.plot_camera(ax, cam2world=cam2world_right, M=intrinsic_matrix, 
                  sensor_size=sensor_size, 
                  virtual_image_distance=virtual_image_distance, 
                  color='green')

# Plot cameras
plot_cameras(ax)

# Process each frame
for i in range(num_images):
    left_img = cv2.imread(left_images[i])
    right_img = cv2.imread(right_images[i])
    
    frame_labels = labels.get(i, [])
    
    points_3d = []
    track_ids = []
    
    for obj in frame_labels:
        track_id = obj['track_id']
        left, top, right, bottom = map(int, obj['bbox'])  # Convert bbox coordinates to integers for drawing
        
        # Predicted point in the left image (center of the bounding box)
        predicted_point_left = ((left + right) // 2, (top + bottom) // 2)
        
        # Corresponding point in the right image (block matching)
        disparity = cv2.matchTemplate(right_img, left_img[top:bottom, left:right], cv2.TM_SQDIFF)
        min_val, _, min_loc, _ = cv2.minMaxLoc(disparity)
        point_right = (min_loc[0] + (right - left) // 2, top + (bottom - top) // 2)
        
        # Prepare points for triangulation (must be in homogeneous coordinates)
        pts_left = np.array([predicted_point_left[0], predicted_point_left[1], 1.0])
        pts_right = np.array([point_right[0], point_right[1], 1.0])
        
        # Triangulate points to get 3D coordinates
        if use_custom_triangulation:
            #points_3D = custom_triangulate(P_left, P_right, pts_left, pts_right)
            points_3D = my_triangulate(T, avg_focal_length_02, avg_focal_length_03, pts_left, pts_right)
            print("points_3D", points_3D)
        else:
            points_4D = cv2.triangulatePoints(P_left, P_right, pts_left[:2].reshape(2, 1), pts_right[:2].reshape(2, 1))
            points_3D = points_4D[:3] / points_4D[3]  # Convert from homogeneous to 3D
        
        points_3d.append(points_3D.flatten())
        track_ids.append(track_id)
        
        # Draw the predicted bounding box in red
        cv2.rectangle(left_img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(left_img, str(track_id), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw bounding box on the right image
        right_bbox_left = min_loc[0]
        right_bbox_top = top
        right_bbox_right = min_loc[0] + (right - left)
        right_bbox_bottom = bottom
        cv2.rectangle(right_img, (right_bbox_left, right_bbox_top), (right_bbox_right, right_bbox_bottom), (255, 255, 255), 2)
        cv2.putText(right_img, str(track_id), (right_bbox_left, right_bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Update 3D plot
    ax.cla()
    plot_cameras(ax)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 20])
    for point, track_id in zip(points_3d, track_ids):
        ax.scatter(point[0], point[1], point[2], label=f'ID {track_id}')
        ax.text(point[0], point[1], point[2], f'{track_id}', color='red')
    
    plt.draw()
    plt.pause(0.001)
    
    # Stack images vertically
    stacked_img = np.vstack((left_img, right_img))

    # Display stacked images
    cv2.imshow('Stacked Images', stacked_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()