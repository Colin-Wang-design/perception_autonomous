import os
import logging
import datetime
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import CustomValidationDataset, BDD100KDataset
from utils import collate_fn, CATEGORY_TO_LABEL
from train import train_and_validate
from test import test_model
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler()
    ]
)

# Argument parsing
parser = argparse.ArgumentParser(description="Train or test the Faster R-CNN model.")
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
parser.add_argument('--model-path', type=str, help="Path to the trained model for testing")
parser.add_argument('--num-classes', type=int, default=len(CATEGORY_TO_LABEL) + 1, help="Number of classes including background")
args = parser.parse_args()

# Generate a timestamp for the output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'/dtu/blackhole/1b/203515/perception/data/output/{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Define transformations for training and validation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet mean and std
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=mean, std=std)  # Normalize with ImageNet mean and std
])

if args.mode == 'train':
    # Paths to your datasets
    train_images_dir = "/dtu/blackhole/1b/203515/perception/data/bdd100k/images/100k/train"
    train_annotations_file = "/dtu/blackhole/1b/203515/perception/data/bdd100k/labels/labels/det_20/det_train.json"

    # Paths to your validation datasets
    val_images_dir_seq1 = "/dtu/blackhole/1b/203515/perception/data/rect/seq_01/image_02/data"
    val_labels_file_seq1 = "/dtu/blackhole/1b/203515/perception/data/rect/seq_01/labels.txt"

    val_images_dir_seq2 = "/dtu/blackhole/1b/203515/perception/data/rect/seq_02/image_02/data"
    val_labels_file_seq2 = "/dtu/blackhole/1b/203515/perception/data/rect/seq_02/labels.txt"

    # Initialize the datasets for both sequences
    # For seq_01 (filenames like '000000.png')
    val_dataset_seq1 = CustomValidationDataset(
        val_images_dir_seq1,
        val_labels_file_seq1,
        transforms=val_transforms,  # Use validation transforms
        filename_format="{:06d}.png"  # Six digits
    )

    # For seq_02 (filenames like '0000000000.png')
    val_dataset_seq2 = CustomValidationDataset(
        val_images_dir_seq2,
        val_labels_file_seq2,
        transforms=val_transforms,  # Use validation transforms
        filename_format="{:010d}.png"  # Ten digits
    )

    # Combine the datasets
    val_dataset = ConcatDataset([val_dataset_seq1, val_dataset_seq2])

    # Initialize the datasets
    train_dataset = BDD100KDataset(train_images_dir, train_annotations_file, transforms=train_transforms, limit=1000)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = len(CATEGORY_TO_LABEL) + 1  # Add 1 for background

    # Replace the classifier with a new one for our dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Optimizer and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Train and validate the model
    num_epochs = 20
    trained_model = train_and_validate(train_loader, val_loader, model, optimizer, num_epochs, output_dir, mean, std)

    print("Training and validation complete!")

elif args.mode == 'test':
    if not args.model_path:
        raise ValueError("Please provide the path to the trained model using --model-path")

    # Paths to your test dataset
    test_images_dir = "/dtu/blackhole/1b/203515/perception/data/rect/seq_03/image_02/data"

    # Run the test routine
    test_model(args.model_path, test_images_dir, output_dir, mean, std, args.num_classes)