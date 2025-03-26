import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on Food-11 dataset")
    parser.add_argument('--train_folder', type=str, default="./data/train/", help="Path to training dataset folder")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--save_path', type=str, default="./models/finetuned_resnet50.pth", help="Path to save trained model")
    
    return parser.parse_args()

def get_data_loaders(train_folder, batch_size):
    """Load dataset and create training & validation data loaders."""
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training folder '{train_folder}' not found.")
    
    # Load ImageNet pretrained weights and transform
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    
    # Load full dataset
    full_dataset = ImageFolder(root=train_folder, transform=transform)
    
    # Split dataset into 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset Loaded: {len(train_dataset)} train images, {len(val_dataset)} validation images")
    return train_loader, val_loader

def build_model(num_classes=11):
    """Initialize and modify the ResNet50 model."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the fully connected (FC) layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Unfreeze FC layers for training
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, device, epochs, learning_rate, save_path):
    """Train and validate the model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nTraining Started...\n")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = (correct / total) * 100
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        val_acc = (correct_val / total_val) * 100
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Model saved at: {save_path}")

def main():
    """Main function to run the training pipeline."""
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get DataLoaders
    train_loader, val_loader = get_data_loaders(args.train_folder, args.batch_size)
    
    # Build model
    model = build_model()
    
    # Train model
    train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.save_path)

if __name__ == "__main__":
    main()
