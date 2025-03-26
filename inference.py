import os
import torch
import argparse
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
model_dir = os.getenv("FINETUNE_MODEL_DIR")

# Class labels
CLASS_LABELS = [
    'apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
    'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi'
]

# Loading preprocess of resnet50 model
weights = ResNet50_Weights.IMAGENET1K_V2
transform = weights.transforms()

# Loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_dir, weights_only=False, map_location=device)
model.eval()


# Inference Function
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        image = transform(image)  # Apply transformations
        image = image.unsqueeze(0).to(device)  # Add batch dimension & move to device

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)  # Get predicted class index

        return CLASS_LABELS[predicted_class.item()]  # Return class label
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Main function to process all images in a folder
def batch_inference(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    results = {}
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        if not os.path.isfile(image_path):
            continue  # Skip directories or invalid files
        
        predicted_label = predict_image(image_path)
        if predicted_label:
            results[filename] = predicted_label
            print(f"{filename} -> {predicted_label}") 

    # Save results to a text file
    output_file = os.path.join(output_folder, "predictions.txt")
    with open(output_file, "w") as f:
        for filename, label in results.items():
            f.write(f"{filename}: {label}\n")

    print(f"\nPredictions saved to: {output_file}")

# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Image Inference with ResNet-50")
    parser.add_argument("--input_folder", type=str, default="inputs/", help="Path to input image folder")
    parser.add_argument("--output_folder", type=str, default="results/", help="Path to save predictions")

    args = parser.parse_args()
    batch_inference(args.input_folder, args.output_folder)
