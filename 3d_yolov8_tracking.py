import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Keep existing camera calibration
K_left = np.array([[9.569475e+02, 0.000000e+00, 6.939767e+02], 
                   [0.000000e+00, 9.522352e+02, 2.386081e+02], 
                   [0.000000e+00, 0.000000e+00, 1.000000e+00]])
K_right = np.array([[9.011007e+02, 0.000000e+00, 6.982947e+02], 
                    [0.000000e+00, 8.970639e+02, 2.377447e+02], 
                    [0.000000e+00, 0.000000e+00, 1.000000e+00]])

R = np.array([[0.99947832, 0.02166116, -0.02395787],
              [-0.02162283, 0.99976448, 0.00185789],
              [0.02399247, -0.00133888, 0.99971125]])
T = np.array([-0.53552388, 0.00666445, -0.01007482])

# Define projection matrices
P_left = np.dot(K_left, np.hstack((np.eye(3), np.zeros((3, 1)))))
P_right = np.dot(K_right, np.hstack((R, T.reshape(3, 1))))

class KalmanTracker:
    def __init__(self, initial_bbox):
        # State: [x, y, dx, dy, ddx, ddy]
        self.state = np.array([[initial_bbox[0]], [initial_bbox[1]], [0], [0], [0], [0]])
        self.P = np.eye(6) * 1000
        self.F = np.array([[1, 0, 1, 0, 0.5, 0],
                          [0, 1, 0, 1, 0, 0.5],
                          [0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]])
        self.R = np.eye(2) * 1
        self.I = np.eye(6)
        self.last_prediction = np.array([initial_bbox[0], initial_bbox[1]])
        self.last_bbox_size = [initial_bbox[2] - initial_bbox[0], 
                             initial_bbox[3] - initial_bbox[1]]

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)
        self.last_prediction = self.state[:2].flatten()
        return self.last_prediction

    def update(self, measurement=None):
        """Update Kalman filter state using measurement or last prediction"""
        if measurement is None:
            Z = self.last_prediction.reshape(2, 1)
        else:
            Z = np.array(measurement).reshape(2, 1)
            
        y = Z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)

def setup_3d_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

def triangulate_points(point_left, point_right):
    point_4d = cv2.triangulatePoints(P_left, P_right,
                                   np.array(point_left).reshape(2, 1),
                                   np.array(point_right).reshape(2, 1))
    point_3d = point_4d[:3] / point_4d[3]
    return point_3d.flatten()

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
    cam2world_right = pt.transform_from_pq([T[0], T[1], T[2], 0, 0, 0, 1])
    pc.plot_camera(ax, cam2world=cam2world_right, M=intrinsic_matrix, 
                  sensor_size=sensor_size, 
                  virtual_image_distance=virtual_image_distance, 
                  color='green')

def update_3d_plot(ax, points_3d, track_ids):
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 19])
    
    # Plot cameras
    plot_cameras(ax)
    
    # Plot tracked objects
    for point, track_id in zip(points_3d, track_ids):
        if -10 <= point[0] <= 10 and -10 <= point[1] <= 10 and -1 <= point[2] <= 19:
            ax.scatter(point[0], point[1], point[2], c='red', marker='o')
            ax.text(point[0], point[1], point[2], f'ID:{track_id}', color='red')
    
    plt.draw()
    plt.pause(0.001)

def draw_detections(img, detections, predictions, trackers, lost_trackers):
    """Draw bounding boxes and IDs on image"""
    # Draw current detections
    if detections.boxes is not None:
        for det in detections.boxes:
            box = det.xyxy[0].cpu().numpy()
            track_id = int(det.id) if det.id is not None else -1
            
            # Current detection box (white)
            cv2.rectangle(img, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (255, 255, 255), 2)
            
            # Add ID and class
            label = f'ID:{track_id}'
            cv2.putText(img, label, 
                       (int(box[0]), int(box[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 2)
    
    # Draw predictions for both visible and lost objects
    for track_id, pred in predictions.items():
        if track_id in trackers:
            tracker = trackers[track_id]
            color = (255, 0, 0)  # Blue for active tracks
        elif track_id in lost_trackers:
            tracker = lost_trackers[track_id]
            color = (0, 0, 255)  # Red for lost tracks
        else:
            continue

        # Draw predicted box with same dimensions as last detection
        w, h = tracker.last_bbox_size
        cv2.rectangle(img,
                     (int(pred[0] - w/2), int(pred[1] - h/2)),
                     (int(pred[0] + w/2), int(pred[1] + h/2)),
                     color, 2)
        
        # Add predicted ID
        cv2.putText(img, f'ID:{track_id}(pred)', 
                   (int(pred[0] - w/2), int(pred[1] - h/2 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # sequence_path = '/Users/maxbrazhnyy/GitHub/34759_pfas/Project/34759_final_project_raw/seq_02'
    sequence_path = '/Users/maxbrazhnyy/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Faculdade/DTU/Autonoums Systems/S3_Fall24/34759 Perception for Autonomous Systems/Project/34759_final_project_rect/seq_01'
    left_images = sorted(glob.glob(f'{sequence_path}/image_02/data/*.png'))
    right_images = sorted(glob.glob(f'{sequence_path}/image_03/data/*.png'))
    print('Found', len(left_images), 'images')
    
    # Initialize trackers dictionary
    trackers = {}
    lost_trackers = {}
    max_lost_frames = 10
    
    fig, ax = setup_3d_plot()
    
    for i in range(len(left_images)):
        left_img = cv2.imread(left_images[i])
        right_img = cv2.imread(right_images[i])
        
        # Add frame number
        cv2.putText(left_img, f"Left Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(right_img, f"Right Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Run YOLOv8 detection
        left_results = model.track(left_img, classes=[0,2,3],conf=0.3, persist=True)[0]
        right_results = model.track(right_img, classes=[0,2,3],conf=0.3, persist=True)[0]
        
        points_3d = []
        track_ids = []
        predictions = {}
        detected_ids = set()
        
        # Process current detections
        if left_results.boxes is not None:
            for left_box in left_results.boxes:
                box = left_box.xyxy[0].cpu().numpy()
                left_center = left_box.xywh[0][:2].cpu().numpy()
                cls = int(left_box.cls)
                track_id = int(left_box.id) if left_box.id is not None else -1
                
                # Process detection and update tracker
                if track_id != -1:
                    # Initialize or update tracker
                    if track_id not in trackers:
                        trackers[track_id] = KalmanTracker(box)
                    
                    tracker = trackers[track_id]
                    tracker.last_bbox_size = [box[2] - box[0], box[3] - box[1]]
                    tracker.update(left_center)
                    predicted_pos = tracker.predict()
                    predictions[track_id] = predicted_pos
                    detected_ids.add(track_id)
                    
                    # If object was previously lost, remove from lost_trackers
                    if track_id in lost_trackers:
                        del lost_trackers[track_id]
        
        # Update lost trackers
        for track_id in list(trackers.keys()):
            if track_id not in detected_ids:
                tracker = trackers[track_id]
                # Continue predicting
                tracker.update(None)
                predicted_pos = tracker.predict()
                predictions[track_id] = predicted_pos
                
                if track_id not in lost_trackers:
                    lost_trackers[track_id] = {'frames_lost': 0}
                lost_trackers[track_id]['frames_lost'] += 1
                
                if lost_trackers[track_id]['frames_lost'] > max_lost_frames:
                    del trackers[track_id]
                    del lost_trackers[track_id]
                else:
                    points_3d.append([predicted_pos[0], predicted_pos[1], 0])
                    track_ids.append(track_id)
        
        # Draw detections and predictions
        draw_detections(left_img, left_results, predictions, trackers, lost_trackers)
        draw_detections(right_img, right_results, predictions, trackers, lost_trackers)
        
        # Stack images vertically
        stacked_img = np.vstack((left_img, right_img))
        
        # Update 3D visualization
        update_3d_plot(ax, points_3d, track_ids)
        
        # Show stacked images
        cv2.imshow('Stereo View', stacked_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()