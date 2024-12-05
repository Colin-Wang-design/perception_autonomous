import torch
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import draw_bounding_boxes, collate_fn, CATEGORY_TO_LABEL, LABEL_TO_CATEGORY, denormalize
from dataset import CustomTestDataset, CustomValidationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from collections import defaultdict

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

def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    try:
        # Try to load the model on GPU
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device('cuda'))
        print("Model loaded on GPU")
    except RuntimeError as e:
        print(f"GPU loading failed: {e}")
        # Fall back to CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(torch.device('cpu'))
        print("Model loaded on CPU")
    
    model.eval()
    return model

def run_inference_and_tracking(model, dataloader, deepsort, device, output_dir, mean, std, confidence_threshold=0.5):
    model.eval()
    
    # Determine frame size from the first valid image in the dataloader
    first_batch = next(iter(dataloader))
    first_image = next((img for img in first_batch if img is not None), None)
    if first_image is None:
        raise ValueError("No valid images found in the dataloader")
    frame_size = (first_image.shape[2], first_image.shape[1])  # (width, height)
    
    # Initialize VideoWriter
    video_path = os.path.join(output_dir, "tracking_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    frame_rate = 10  # Adjust as needed
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, frame_size)
    
    trackers = {}
    lost_trackers = {}
    max_lost_frames = 10
    
    total_frames = len(dataloader)
    
    with torch.no_grad():
        for idx, images in enumerate(dataloader):
            # Filter out None images
            images = [img for img in images if img is not None]
            if not images:
                logging.error(f"No valid images in batch {idx}")
                continue
            
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Prepare detections for DeepSORT
                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    if score > confidence_threshold:  # Confidence threshold
                        x1, y1, x2, y2 = box
                        if x2 > x1 and y2 > y1:  # Ensure the box is valid
                            detections.append((box, score, label))
                
                # Update DeepSORT tracker
                frame = images[i].cpu()
                frame = denormalize(frame, mean, std)  # Denormalize the frame
                frame = frame.numpy().transpose(1, 2, 0)  # Convert to HWC format
                frame = (frame * 255).astype(np.uint8)  # Convert to uint8
                frame = np.ascontiguousarray(frame)  # Ensure the array is contiguous
                tracks = deepsort.update_tracks(detections, frame=frame)
                
                # Ensure LABEL_TO_CATEGORY is defined correctly
                LABEL_TO_CATEGORY = {
                    0: 'person',
                    1: 'bicycle',
                    2: 'car',
                    # Add other mappings as needed
                }

                # Visualize or save tracking results
                category_counts = defaultdict(int)
                predictions = {}
                detected_ids = set()
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_tlbr()  # Get bounding box in (top, left, bottom, right) format
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Ensure the box is valid
                        # Debug print to check track.label
                        print(f"Track ID: {track_id}, Label: {getattr(track, 'label', 'No label')}")

                        # Ensure track.label exists and is correctly mapped
                        if hasattr(track, 'label'):
                            category = LABEL_TO_CATEGORY.get(track.label, 'unknown')
                        else:
                            category = 'unknown'

                        # Debug print to check category
                        print(f"Category: {category}")

                        category_counts[category] += 1
                        sub_id = category_counts[category]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"{category} {sub_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Initialize or update Kalman tracker
                        if track_id not in trackers:
                            trackers[track_id] = KalmanTracker(bbox)

                        tracker = trackers[track_id]
                        tracker.last_bbox_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        tracker.update([bbox[0], bbox[1]])
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
                
                # Draw classification bounding boxes
                for box, label, score in zip(boxes, labels, scores):
                    if score > confidence_threshold:  # Confidence threshold
                        x1, y1, x2, y2 = box
                        if x2 > x1 and y2 > y1:  # Ensure the box is valid
                            category = LABEL_TO_CATEGORY.get(label, 'unknown')  # Handle unknown labels
                            category_counts[category] += 1
                            sub_id = category_counts[category]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(frame, f"{category} {sub_id}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Write the frame to the video
                video_writer.write(frame)
                
                # Save the image with bounding boxes
                output_path = os.path.join(output_dir, f"tracked_image_{idx}.png")
                cv2.imwrite(output_path, frame)

                # Draw bounding boxes using the utility function
                draw_bounding_boxes(images[i].cpu(), output, None, idx, output_dir, mean, std, epoch=0, phase="test")
                
                # Print successfully processed frame
                print(f"Successfully processed frame {idx + 1}/{total_frames} ({(idx + 1) / total_frames * 100:.2f}%)")
            
            # Clear cache after each batch
            torch.cuda.empty_cache()
    
    # Release the VideoWriter
    video_writer.release()

def test_model(model_path, test_images_dir, output_dir, mean, std, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_path, num_classes)
    model.to(device)

    # Initialize DeepSORT
    deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=None)

    # Define transformations for testing
    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet mean and std
    ])

    # Initialize the test dataset
    test_dataset = CustomTestDataset(
        test_images_dir,
        transforms=test_transforms,
        filename_format="{:06d}.png"  # Adjust if necessary
    )

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    # Run inference and tracking
    run_inference_and_tracking(model, test_loader, deepsort, device, output_dir, mean, std, confidence_threshold=0.8)
