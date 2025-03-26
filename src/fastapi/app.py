import os
import io
import torch
import uvicorn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException


# Initialize FastAPI app
app = FastAPI(title="Food-11 Classification API", version="1.0")


# Define class labels
CLASS_LABELS = [
    'apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
    'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi'
]


# Load model and preprocessing transforms
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load("./models/finetuned_resnet50.pth", weights_only=False, map_location=device)
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


model, device = load_model()
weights = ResNet50_Weights.IMAGENET1K_V2
transform = weights.transforms()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes into a preprocessed tensor suitable for model input."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image)
        return image.unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to classify an uploaded food image."""
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
        return {"predicted_class": CLASS_LABELS[predicted_class.item()]}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# Run API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
