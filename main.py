import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
from io import BytesIO

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/plant_disease_model_phase2 (1).h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)
model.compile()

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to preprocess the image and make predictions
def predict_disease(model, class_indices, image):
    # Preprocess the image (resize to match model input size and normalize)
    img = image.resize((224, 224))  # Resize image to match model input shape (224x224 assumed)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with highest probability
    predicted_label = class_indices[str(predicted_class)]
    confidence = np.max(prediction) * 100  
    return predicted_label, confidence

"""
Streamlit web application for plant disease detection
"""
st.title("🌿Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect disease")
    
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None: 
    # Open and display the uploaded image
    try:
        # Use BytesIO to read the uploaded image file
        image = Image.open(BytesIO(uploaded_file.read()))

        # Debugging: Display the image size and format
        st.write(f"Image size: {image.size}")
        st.write(f"Image format: {image.format}")

        # Display the uploaded image in the Streamlit app
        st.image(image, caption='Uploaded Image', use_column_width=True)

    except Exception as e:
        st.error(f"Error opening image: {e}")

    # Make prediction when the 'Predict' button is clicked
    # Make prediction
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            disease_name, confidence = predict_disease(model, class_indices, image)
            
        # Display results
        st.success(f"Prediction: {disease_name}")
        st.info(f"Confidence: {confidence:.2f}%")
            
        # Display treatment recommendations based on disease
        st.subheader("Recommended Treatment:")
        # This would be expanded with actual treatments for each disease
        # For now, a placeholder message
        st.write("Please consult an agricultural expert for specific treatment options for this disease.")