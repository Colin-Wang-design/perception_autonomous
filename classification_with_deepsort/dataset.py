import os
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
import json

CATEGORY_TO_LABEL = {
    "other_person": 1,
    "pedestrian": 1, 
    "cyclist": 2,
    "bycicle": 2,
    "rider": 2,
    "other_vehicle": 3,
    "car": 3,
}
LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_LABEL.items()}

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

class CustomValidationDataset(Dataset):
    def __init__(self, images_dir, labels_file=None, transforms=None, filename_format="{:010d}.png"):
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.transforms = transforms
        self.filename_format = filename_format
        self.image_files = sorted(os.listdir(images_dir))
        
        if labels_file:
            self.annotations = self.load_annotations(labels_file)
        else:
            self.annotations = None

    def load_annotations(self, labels_file):
        return self.parse_labels(labels_file)

    def parse_labels(self, labels_file):
        raw_annotations = parse_labels(labels_file)
        annotations = {}
        total_objects = 0
        for frame, objects in raw_annotations.items():
            image_id = self.filename_format.format(frame)
            annotations[image_id] = []
            for obj in objects:
                bbox = obj['bbox']
                object_type = obj['object_type'].lower()
                if object_type in CATEGORY_TO_LABEL:
                    label = CATEGORY_TO_LABEL[object_type]
                    annotations[image_id].append({'bbox': bbox, 'label': label})
                    total_objects += 1
                else:
                    logging.warning(f"Unknown object type: {object_type}")
        
        logging.info(f"Parsed {len(annotations)} images with {total_objects} total objects")
        logging.info(f"Sample image_ids: {list(annotations.keys())[:5]}")
        return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
        
        if self.annotations:
            # Use the image_file directly since it matches how we stored it
            if image_file in self.annotations:
                targets = self.annotations[image_file]
                boxes = [target['bbox'] for target in targets]
                labels = [target['label'] for target in targets]
                target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64)
                }
                logging.info(f"Found {len(boxes)} boxes for image {image_file}")
                return image, target
            else:
                logging.warning(f"No annotations found for image {image_file}")
                return image, {'boxes': torch.tensor([]), 'labels': torch.tensor([])}

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None, limit=None):
        logging.info(f"Initializing BDD100KDataset with {images_dir} and {annotations_file}")
        self.images_dir = images_dir
        self.transforms = transforms

        try:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            logging.info(f"Loaded {len(self.annotations)} annotations from {annotations_file}.")
            if limit:
                self.annotations = self.annotations[:limit]
                logging.info(f"Limited dataset to {len(self.annotations)} annotations.")
        except Exception as e:
            logging.error(f"Error loading annotations file: {e}")
            raise e

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        if 'labels' not in annotation:
            logging.warning(f"Missing 'labels' key in annotation at index {idx}: {annotation}")
            return None

        if not annotation['labels']:
            logging.warning(f"Empty 'labels' for image at index {idx}: {annotation}")
            return None

        img_path = os.path.join(self.images_dir, annotation['name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logging.error(f"Image not found at path: {img_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None

        boxes = []
        labels = []
        for obj in annotation['labels']:
            if obj['category'] not in CATEGORY_TO_LABEL:
                continue
            if 'box2d' in obj:
                box = obj['box2d']
                boxes.append([box['x1'], box['y1'], box['x2'], box['y2']])
                labels.append(CATEGORY_TO_LABEL[obj['category']])
            else:
                logging.warning(f"Missing 'box2d' key in object at index {idx}: {obj}")

        if not boxes:
            logging.warning(f"No valid bounding boxes for image at index {idx}: {annotation}")
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        logging.info(f"Successfully processed image and annotations for index {idx}.")
        return image, target