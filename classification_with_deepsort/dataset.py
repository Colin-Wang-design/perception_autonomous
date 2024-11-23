import json
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import logging

# Category mapping (only cars, pedestrians, and cyclists)
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

class CustomValidationDataset(Dataset):
    def __init__(self, images_dir, labels_file=None, transforms=None, filename_format="{:06d}.png"):
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
        annotations = {}
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                image_id = parts[0]
                bbox = list(map(float, parts[1:5]))
                label = CATEGORY_TO_LABEL.get(parts[5], 0)
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append({'bbox': bbox, 'label': label})
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
            image_id = os.path.splitext(image_file)[0]
            targets = self.annotations.get(image_id, [])
            boxes = [target['bbox'] for target in targets]
            labels = [target['label'] for target in targets]
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
            return image, target
        else:
            return image

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

        # logging.info(f"Successfully processed image and annotations for index {idx}.")
        return image, target