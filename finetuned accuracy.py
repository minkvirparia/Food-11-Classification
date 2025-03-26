import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score


def load_model(model_path, device):
    """Loads the fine-tuned ResNet-50 model."""
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()  # Set model to evaluation mode
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def load_test_data(data_path, transform):
    """Loads the test dataset."""
    try:
        test_dataset = ImageFolder(root=data_path, transform=transform)
        print(f"Loaded test dataset with {len(test_dataset)} images.")
        return test_dataset
    except Exception as e:
        raise RuntimeError(f"Error loading test dataset: {e}")


def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset and calculates accuracy."""
    model.to(device)
    model.eval()  # Ensure model is in evaluation mode

    all_preds, all_labels = [], []

    print("Evaluating model on test dataset...")

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device

            # Get model predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())  # Convert to NumPy
            all_labels.extend(labels.cpu().numpy())  # Convert to NumPy

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    """Main function to load model, dataset, and evaluate accuracy."""
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformation for test images (same preprocessing as training)
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()

    # Load model
    model = load_model("./finetuned_resnet50.pth", device)

    # Load test dataset 
    test_dataset = load_test_data("D:/Mind Inventory Task/food11 dataset/test/", transform)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
