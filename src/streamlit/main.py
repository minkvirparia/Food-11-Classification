import io
import requests
import streamlit as st
from PIL import Image

# FastAPI endpoint URL
API_URL = "http://localhost:8000/predict/"

def send_image_for_prediction(image):
    """Sends the uploaded image to the FastAPI server for classification."""
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        
        response = requests.post(API_URL, files={"file": image_bytes})
        response.raise_for_status()
        return response.json().get("predicted_class", "Unknown")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def main():
    """Streamlit UI for Food-11 Image Classification."""
    st.title("🍕 Food-11 Classification")
    st.write("Upload a food image to get its classification.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Submit", use_container_width=True):
            predicted_class = send_image_for_prediction(image)
            st.success(f"🍽️ This image most likely is **{predicted_class}**")

if __name__ == "__main__":
    main()
