import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
from matplotlib.animation import FuncAnimation

# Step 1: Read the file and store the information
data = defaultdict(lambda: {'left': {}, 'right': {}})

with open('TRACKED_CENTERS3.TXT', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        camera, frame, track_id, x_center, y_center = row
        frame = int(frame)
        track_id = int(track_id)
        x_center = float(x_center)
        y_center = float(y_center)
        data[frame][camera][track_id] = (x_center, y_center)

# Step 2: Identify and remove track IDs not present in both cameras
for frame in data.keys():
    left_ids = set(data[frame]['left'].keys())
    right_ids = set(data[frame]['right'].keys())
    common_ids = left_ids & right_ids

    # Remove track IDs not present in both cameras
    data[frame]['left'] = {k: v for k, v in data[frame]['left'].items() if k in common_ids}
    data[frame]['right'] = {k: v for k, v in data[frame]['right'].items() if k in common_ids}

# Camera parameters copied from calib_cam_to_cam.txt
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

# Calculate the average focal lengths for each camera
avg_focal_length_02 = (f_x_02 + f_y_02) / 2
avg_focal_length_03 = (f_x_03 + f_y_03) / 2

# Update translation vector T to reflect the 0.54 meters separation
T = np.array([-0.54, 0, 0])

def my_triangulate(T, f_02, f_03, pts_02, pts_03):
    # Calculate the disparity
    disparity = pts_02[0] - pts_03[0]
    if disparity <= 0:
        print(f"Disparity is not positive: {disparity}")

    f = (f_02 + f_03) / 2
    f = f_02
    # Calculate the depth in meters
    depth = abs(T[0]) * f / disparity
    
    # Calculate the 3D point
    X = np.array([
        depth * (pts_02[0] - K_left[0, 2]) / f,
        depth * (pts_02[1] - K_left[1, 2]) / f,
        depth
    ])
    
    return X

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

# Function to update the plot for each frame
def update(frame):
    ax.cla()
    plot_cameras(ax)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([0, 20])
    
    for track_id in data[frame]['left'].keys():
        pts_left = data[frame]['left'][track_id]
        pts_right = data[frame]['right'][track_id]
        X = my_triangulate(T, avg_focal_length_02, avg_focal_length_03, pts_left, pts_right)*10
        print(f"Frame: {frame}, Track ID: {track_id}, 3D Coordinates: X={X[0]}, Y={X[1]}, Z={X[2]}, Camera Frame: Left({pts_left[0]}, {pts_left[1]}), Right({pts_right[0]}, {pts_right[1]})")
        ax.scatter(X[0], X[1], X[2], label=f'ID {track_id}')
        ax.text(X[0], X[1], X[2], f'{track_id}', color='red')

# Create animation
frames = sorted(data.keys())
ani = FuncAnimation(fig, update, frames=frames, repeat=False)

plt.show()
