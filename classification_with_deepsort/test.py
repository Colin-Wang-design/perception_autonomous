import torch
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import draw_bounding_boxes, collate_fn, CATEGORY_TO_LABEL, LABEL_TO_CATEGORY, denormalize
from dataset import CustomValidationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from collections import defaultdict

def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference_and_tracking(model, dataloader, deepsort, device, output_dir, mean, std, confidence_threshold=0.5):
    model.eval()
    
    # Determine frame size from the first image in the dataloader
    first_batch = next(iter(dataloader))
    first_image = first_batch[0] if isinstance(first_batch, list) else first_batch
    frame_size = (first_image.shape[2], first_image.shape[1])  # (width, height)
    
    # Initialize VideoWriter
    video_path = os.path.join(output_dir, "tracking_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    frame_rate = 10  # Adjust as needed
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, frame_size)
    
    with torch.no_grad():
        for idx, images in enumerate(dataloader):
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
                
                # Visualize or save tracking results
                category_counts = defaultdict(int)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_tlbr()  # Get bounding box in (top, left, bottom, right) format
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Ensure the box is valid
                        category = LABEL_TO_CATEGORY.get(track.label, 'unknown') if hasattr(track, 'label') else 'unknown'
                        category_counts[category] += 1
                        sub_id = category_counts[category]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"{category} {sub_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
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
                draw_bounding_boxes(images[i].cpu(), output, None, idx, output_dir, mean, std, epoch=0)
            
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
    test_dataset = CustomValidationDataset(
        test_images_dir,
        labels_file=None,  # No labels for the test set
        transforms=test_transforms,
        filename_format="{:06d}.png"  # Adjust if necessary
    )

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    # Run inference and tracking
    run_inference_and_tracking(model, test_loader, deepsort, device, output_dir, mean, std, confidence_threshold=0.8)