import os
import torch
from PIL import Image
import streamlit as st
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# Load model once on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the trained model once
@st.cache_resource
def load_model():
    model = torch.load("./models/finetuned_resnet50.pth", weights_only=False, map_location=device)
    model.eval()
    return model

model = load_model()


# Define class labels
CLASS_LABELS = [
    "apple_pie", "cheesecake", "chicken_curry", "french_fries", "fried_rice", 
    "hamburger", "hot_dog", "ice_cream", "omelette", "pizza", "sushi"
]

# Define image preprocessing
weights = ResNet50_Weights.IMAGENET1K_V2
transform = weights.transforms()


# Streamlit UI
st.title("🍕 Food-11 Classification")
st.write("Please upload a food image")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Submit", use_container_width=True):
        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASS_LABELS[predicted.item()]

        # Display the result
        st.success(f"🍽️ This image most likely is **{predicted_class}**")
