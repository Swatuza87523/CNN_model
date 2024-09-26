import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model_path = 'E:\\Deep Learning\\t.h5'
model = load_model(model_path)

# Define the class names
class_names = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

def preprocess_image(img):
    img = img.resize((256, 256))  # Set your target dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if you did this during training
    return img_array

# Streamlit interface
st.title("Brain Tumor Classification")
st.write("Upload a brain MRI image to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)  # Load image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    preprocessed_img = preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]

    # Display prediction
    st.write(f"**Predicted Class:** {predicted_class_name}")

# Run with: streamlit run app.py
