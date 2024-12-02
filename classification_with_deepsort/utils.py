import csv
import logging
import torch
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import numpy as np
import pandas as pd

# Category mapping (only cars, pedestrians, and cyclists)
CATEGORY_TO_LABEL = {
    "pedestrian": 1, 
    "other_person": 1,
    "cyclist": 2,
    "rider": 2,
    "bycicle": 2,
    "car": 3,
    "other_vehicle": 3,
}
LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_LABEL.items()}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        logging.error("All samples in the batch were skipped. Check your dataset!")
        raise ValueError("Empty batch encountered. Please debug your dataset.")
    logging.info(f"Batch size: {len(batch)}")
    return tuple(zip(*batch))

def calculate_metrics(predictions, targets, iou_threshold=0.8, output_dir=None, epoch=None):
    metrics = {}
    class_metrics = {}
    
    logging.info(f"Calculating metrics for {len(predictions)} predictions")
    logging.info(f"First prediction boxes: {predictions[0]['boxes'].shape}")
    logging.info(f"First prediction scores: {predictions[0]['scores']}")
    logging.info(f"First target boxes: {targets[0]['boxes'].shape}")
    
    for i in range(len(predictions)):
        pred_boxes = predictions[i]['boxes']
        pred_labels = predictions[i]['labels']
        pred_scores = predictions[i]['scores']
        
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']
        
        # Log shapes and unique labels
        logging.info(f"Sample {i}:")
        logging.info(f"Pred boxes: {pred_boxes.shape}, labels: {torch.unique(pred_labels)}")
        logging.info(f"GT boxes: {gt_boxes.shape}, labels: {torch.unique(gt_labels)}")
        
        conf_mask = pred_scores > 0.8
        pred_boxes = pred_boxes[conf_mask]
        pred_labels = pred_labels[conf_mask]
        pred_scores = pred_scores[conf_mask]
        
        for cls in torch.unique(gt_labels):
            if cls == 0:
                continue
            
            gt_cls_mask = gt_labels == cls
            cls_gt_boxes = gt_boxes[gt_cls_mask]
            
            pred_cls_mask = pred_labels == cls
            cls_pred_boxes = pred_boxes[pred_cls_mask]
            
            if len(cls_gt_boxes) > 0 and len(cls_pred_boxes) > 0:
                ious = box_iou(cls_gt_boxes, cls_pred_boxes)
                
                max_iou, max_idx = torch.max(ious, dim=0)
                matches = max_iou > iou_threshold
                
                # Debugging: Log IoU values and matches
                logging.info(f"IoUs for class {cls}: {ious}")
                logging.info(f"Matches for class {cls}: {matches.sum().item()}")

                if cls not in class_metrics:
                    class_metrics[cls] = {
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'total_gt': len(cls_gt_boxes)
                    }
                
                class_metrics[cls]['true_positives'] += matches.sum().item()
                class_metrics[cls]['false_positives'] += (matches.size(0) - matches.sum().item())
                class_metrics[cls]['false_negatives'] += max(0, len(cls_gt_boxes) - matches.sum().item())
    
    for cls, stats in class_metrics.items():
        precision = (stats['true_positives'] / 
                     (stats['true_positives'] + stats['false_positives'] + 1e-10))
        recall = (stats['true_positives'] / 
                  (stats['true_positives'] + stats['false_negatives'] + 1e-10))
        f1 = (2 * precision * recall) / (precision + recall + 1e-10)
        
        metrics[LABEL_TO_CATEGORY.get(cls, f'Class {cls}')] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_ground_truth': stats['total_gt']
        }
    
    if len(metrics) > 0:
        metrics['mAP'] = sum(metrics[cls]['precision'] for cls in metrics) / len(metrics)
    else:
        metrics['mAP'] = 0.0
    
    # Save metrics to CSV
    if output_dir and epoch is not None:
        csv_file = os.path.join(output_dir, 'metrics.csv')
        fieldnames = ['epoch', 'class', 'precision', 'recall', 'f1_score', 'total_ground_truth']
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            for cls, stats in metrics.items():
                if cls != 'mAP':
                    writer.writerow({
                        'epoch': epoch,
                        'class': cls,
                        'precision': stats['precision'],
                        'recall': stats['recall'],
                        'f1_score': stats['f1_score'],
                        'total_ground_truth': stats['total_ground_truth']
                    })
            writer.writerow({
                'epoch': epoch,
                'class': 'mAP',
                'precision': metrics['mAP'],
                'recall': '',
                'f1_score': '',
                'total_ground_truth': ''
            })
    
    return metrics
def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def draw_bounding_boxes(image, prediction, target, image_index, output_dir, mean, std, epoch, phase):
    confidence_threshold = 0.8  # Adjust as needed

    # Filter out predictions with low confidence
    logging.debug(f"predictions {prediction}" )
    high_conf_idxs = prediction['scores'] > confidence_threshold
    prediction['boxes'] = prediction['boxes'][high_conf_idxs]
    prediction['labels'] = prediction['labels'][high_conf_idxs]
    prediction['scores'] = prediction['scores'][high_conf_idxs]
    
    # Convert PIL image to tensor if necessary
    if isinstance(image, Image.Image):
        image = F.to_tensor(image)
    
    # Denormalize the image
    image = denormalize(image, mean, std)
    
    # Convert tensor to numpy array and clip values between 0 and 1
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Draw predicted bounding boxes in red
    for box, label, score in zip(
        prediction['boxes'].detach().cpu().numpy(),
        prediction['labels'].detach().cpu().numpy(),
        prediction['scores'].detach().cpu().numpy()
    ):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin - 10,
            f"{LABEL_TO_CATEGORY.get(label, 'Unknown')} {score:.2f}",
            color='r', fontsize=8, weight='bold'
        )

    # Draw ground truth bounding boxes in green (if available)
    if target is not None:
        for box, label in zip(
            target['boxes'].detach().cpu().numpy(),
            target['labels'].detach().cpu().numpy()
        ):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin, ymax + 10,
                LABEL_TO_CATEGORY.get(label, 'Unknown'),
                color='g', fontsize=8, weight='bold'
            )

    # Remove axis ticks and labels
    plt.axis('off')

    # Adjust output directory to include phase and epoch
    if phase == 'test':
        output_dir = os.path.join(output_dir, phase)
    else:
        output_dir = os.path.join(output_dir, phase, f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'image_{image_index}.png')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_metrics(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    
    epochs = df['epoch'].unique()
    classes = df['class'].unique()
    
    for cls in classes:
        if cls == 'mAP':
            plt.figure()
            plt.plot(df[df['class'] == cls]['epoch'], df[df['class'] == cls]['precision'], label='mAP')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision (mAP) Over Epochs')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'mAP_over_epochs.png'))
        else:
            plt.figure()
            plt.plot(df[df['class'] == cls]['epoch'], df[df['class'] == cls]['precision'], label='Precision')
            plt.plot(df[df['class'] == cls]['epoch'], df[df['class'] == cls]['recall'], label='Recall')
            plt.plot(df[df['class'] == cls]['epoch'], df[df['class'] == cls]['f1_score'], label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title(f'{cls.capitalize()} Metrics Over Epochs')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{cls}_metrics_over_epochs.png'))