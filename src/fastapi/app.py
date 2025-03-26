import os
import io
import torch
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException


# Load environment variables from .env file
load_dotenv()
model_dir = os.getenv("FINETUNE_MODEL_DIR")


# Initialize FastAPI app
app = FastAPI(title="Food-11 Classification API", version="1.0")


# Define class labels
CLASS_LABELS = [
    'apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
    'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi'
]


# Load model once on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(model_dir, weights_only=False, map_location=device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define image preprocessing
weights = ResNet50_Weights.IMAGENET1K_V2
transform = weights.transforms()

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes into a preprocessed tensor suitable for model input.

    Args:
        image_bytes (bytes): Raw image bytes from user input.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for inference.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure RGB format
        image = transform(image)  # Apply transformations
        return image.unsqueeze(0).to(device)  # Add batch dimension & move to device
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to classify an uploaded food image.

    Args:
        file (UploadFile): Image file uploaded by the user.

    Returns:
        dict: JSON response containing the predicted class label.
    """
    try:
        image_bytes = await file.read()  # Read image bytes
        image_tensor = preprocess_image(image_bytes)  # Preprocess image

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)  # Get predicted class index

        return {"predicted_class": CLASS_LABELS[predicted_class.item()]}

    except HTTPException as he:
        raise he  # Handle bad requests properly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Run API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
