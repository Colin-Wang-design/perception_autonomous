import torch
import logging
import os
from utils import calculate_metrics, draw_bounding_boxes

def train_and_validate(train_loader, val_loader, model, optimizer, num_epochs, output_dir, mean, std, save_examples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Training with", device)
    
    best_map = 0.0
    
    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        
        for images, targets in train_loader:
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
                valid_train_batches += 1
            except Exception as e:
                logging.error(f"Error during training at epoch {epoch}, batch {train_loader.batch_size}: {e}")
                continue
        
        model.eval()
        val_loss = 0.0
        valid_val_batches = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for idx, (images, targets) in enumerate(val_loader):
                try:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    predictions = model(images)

                    images_cpu = [img.cpu() for img in images]
                    predictions_cpu = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
                    targets_cpu = [{k: v.cpu() for k, v in target.items()} for target in targets]

                    val_predictions.extend(predictions_cpu)
                    val_targets.extend(targets_cpu)

                    # Save only a few examples
                    if idx < save_examples:
                        for i in range(len(images_cpu)):
                            image = images_cpu[i]
                            pred = predictions_cpu[i]
                            target = targets_cpu[i]

                            draw_bounding_boxes(image, pred, target, idx * val_loader.batch_size + i, output_dir, mean, std, epoch)
                    
                    # Calculate validation loss
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    valid_val_batches += 1
                except Exception as e:
                    logging.error(f"Error during validation at epoch {epoch}, batch {val_loader.batch_size}: {e}")
                    continue
        
        print(f"Number of predictions: {len(val_predictions)}")
        for i, pred in enumerate(val_predictions):
            print(f"Predictions for image {i}: {pred}")
        
        val_metrics = calculate_metrics(val_predictions, val_targets)
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}")
        if valid_train_batches > 0:
            print(f"Training Loss: {train_loss / valid_train_batches:.4f}")
        else:
            print("Training Loss: N/A (no valid batches)")
        
        if valid_val_batches > 0:
            print(f"Validation Loss: {val_loss / valid_val_batches:.4f}")
        else:
            print("Validation Loss: N/A (no valid batches)")
        
        print("Validation Metrics:")
        for cls, metrics_dict in val_metrics.items():
            if cls == 'mAP':
                print(f"mAP: {metrics_dict:.4f}")
            else:
                print(f"{cls}:")
                for metric_name, value in metrics_dict.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        # Update the learning rate
        if valid_val_batches > 0:
            scheduler.step(val_loss / valid_val_batches)
        
        # Save best model based on mAP
        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            model_save_path = os.path.join(output_dir, f"fasterrcnn_bdd100k_best_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with mAP {best_map:.4f}")
    
    return model