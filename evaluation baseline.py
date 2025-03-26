import os
import torch
import random
import numpy as np
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score


# Load environment variables from .env file
load_dotenv()
data_dir = os.getenv("DATASET_DIR")  # Fetch dataset path from .env

def set_seed(seed=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model():
    set_seed(42)  # Set seed for reproducibility
    
    # Load ResNet-50 model with pre-trained weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    num_classes = 11  # Number of classes in Food-11 dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Update for Food-11 classes
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load Dataset
    preprocess = weights.transforms()
    test_dataset = ImageFolder(root=os.path.join(data_dir, "test"), transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate Model
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\nBaseline Model Evaluation (Before Fine-Tuning):\n")
    print(f"Model: ResNet-50 (Pre-trained on ImageNet)")
    print(f"Dataset: Food-11")
    print(f"Initial Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

evaluate_model()