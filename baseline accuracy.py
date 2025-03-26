import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score


def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(device):
    """Load ResNet-50 model with pre-trained weights and modify the FC layer."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    num_classes = 11
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    
    return model, weights


def load_test_data(weights, batch_size=32):
    """Load test dataset and apply preprocessing."""
    transform = weights.transforms()
    test_dataset = ImageFolder(root="D:/Mind Inventory Task/food11 dataset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def evaluate_model():
    """Evaluate model performance before fine-tuning."""
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = load_model(device)
    test_loader = load_test_data(weights)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\nBaseline Model Evaluation (Before Fine-Tuning):")
    print(f"Model: ResNet-50 (Pre-trained on ImageNet)")
    print(f"Dataset: Food-11")
    print(f"Initial Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    evaluate_model()
