import torch

model_path = '/Users/maxbrazhnyy/Downloads/fasterrcnn_best_epoch_7.pth'

try:
    state_dict = torch.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")