import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("trained_animal_classifier_model.keras")

# Define class names (same order as training)
class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer',
    'Dog', 'Dolphin', 'Elephant', 'Giraffe',
    'Horse', 'Kangaroo', 'Lion', 'Panda',
    'Tiger', 'Zebra'
]

# App UI
st.set_page_config(page_title="Animal Classifier", page_icon="🐾")
st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image and the model will try to predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Prediction: **{predicted_class}**")
