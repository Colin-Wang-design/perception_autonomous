import torch
import logging
import os
from utils import calculate_metrics, draw_bounding_boxes, plot_metrics
from torchvision.transforms import functional as F

def train_and_validate(train_loader, val_loader, model, optimizer, num_epochs, output_dir, mean, std):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Training with %s", device)
    
    best_map = 0.0
    early_stopping_patience = 10
    no_improvement_epochs = 0
    
    # Learning rate scheduler based on validation mAP
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            train_batches += 1

            # Optional: Save training examples
            if train_batches <= 5:  # Save first 5 batches
                model.eval()
                with torch.no_grad():
                    preds = model(images)
                for i in range(len(images)):
                    image = images[i].cpu()
                    pred = {k: v.cpu() for k, v in preds[i].items()}
                    target = {k: v.cpu() for k, v in targets[i].items()}
                    draw_bounding_boxes(image, pred, target, epoch * len(train_loader) + i, output_dir, mean, std, epoch, phase='train')
                model.train()

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for idx, (images, targets) in enumerate(val_loader):
                images = [img.to(device) for img in images]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                outputs = model(images)
                outputs_cpu = [{k: v.cpu() for k, v in output.items()} for output in outputs]

                val_predictions.extend(outputs_cpu)
                val_targets.extend(targets_cpu)

                # Debugging: Log predictions and targets
                logging.info(f"Validation Outputs at batch {idx}:")
                for output in outputs_cpu:
                    logging.info(output)
                logging.info(f"Validation Targets at batch {idx}:")
                for target in targets_cpu:
                    logging.info(target)

                # Call draw_bounding_boxes for all validation images
                for i in range(len(images)):
                    image = images[i].cpu()
                    pred = outputs_cpu[i]
                    target = targets_cpu[i]
                    draw_bounding_boxes(
                        image, pred, target,
                        idx * val_loader.batch_size + i,
                        output_dir, mean, std, epoch,
                        phase='val'
                    )

        val_metrics = calculate_metrics(val_predictions, val_targets, output_dir=output_dir, epoch=epoch)
        current_map = val_metrics.get('mAP', 0.0)
        
        # Print epoch summary
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Training Loss: {avg_train_loss:.4f}")
        logging.info(f"Validation mAP: {current_map:.4f}")

        # Early stopping and model saving
        if current_map > best_map:
            best_map = current_map
            model_save_path = os.path.join(output_dir, f"fasterrcnn_best_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved with mAP {best_map:.4f}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Update the learning rate based on validation mAP
        scheduler.step(current_map)

    # Plot metrics after training is complete
    csv_file = os.path.join(output_dir, 'metrics.csv')
    plot_metrics(csv_file, output_dir)

    return model