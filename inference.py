import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# Class labels
CLASS_LABELS = [
    "apple_pie", "cheesecake", "chicken_curry", "french_fries", "fried_rice",
    "hamburger", "hot_dog", "ice_cream", "omelette", "pizza", "sushi"
]


def load_model(model_path, device):
    """Loads the fine-tuned ResNet-50 model."""
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_image(image_path, transform):
    """Loads and preprocesses an image for inference."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> str:
    """Runs inference on a single image and returns the predicted class label."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)  # Get predicted class index
    return CLASS_LABELS[predicted_class.item()]


def batch_inference(model: torch.nn.Module, input_folder: str, output_folder: str, transform: transforms.Compose, device: torch.device) -> None:
    """Runs inference on all images in a folder and saves results."""
    os.makedirs(output_folder, exist_ok=True)
    results: Dict[str, str] = {}
    
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        if not os.path.isfile(image_path):
            continue  # Skip directories or invalid files
        
        image_tensor = preprocess_image(image_path, transform)
        if image_tensor is not None:
            predicted_label = predict_image(model, image_tensor, device)
            results[filename] = predicted_label
            print(f"{filename} -> {predicted_label}")
    
    # Save results to a text file
    output_file = os.path.join(output_folder, "predictions.txt")
    with open(output_file, "w") as f:
        for filename, label in results.items():
            f.write(f"{filename}: {label}\n")
    
    print(f"\nPredictions saved to: {output_file}")

def main():
    """Main function to handle argument parsing and initiate batch inference."""
    parser = argparse.ArgumentParser(description="Batch Image Inference with ResNet-50")
    parser.add_argument("--model_path", type=str, default="finetuned_resnet50.pth", help="Path to the trained model file")
    parser.add_argument("--input_folder", type=str, default="inputs/", help="Path to input image folder")
    parser.add_argument("--output_folder", type=str, default="results/", help="Path to save predictions")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    
    print("\nStarting inference...\n")
    batch_inference(model, args.input_folder, args.output_folder, transform, device)
    print("\nInference completed successfully!\n")

if __name__ == "__main__":
    main()
