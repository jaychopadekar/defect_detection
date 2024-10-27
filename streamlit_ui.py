import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
try:
    model = tf.keras.models.load_model('defect_detect_model2.h5')
    st.write("Model loaded successfully.")
    
    # Class names (ensure these match your training labels)
    class_names = ['Crack', 'Dent', 'No defect']

except Exception as e:
    st.write("Error loading model:", e)

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to make predictions and provide recommendations
def predict_defect(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Get predicted class and confidence
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Define recommendation based on confidence and predicted class
    if predicted_class in ["Dent", "Crack"]:
        if confidence > 0.8:
            recommendation = "Repair needed"
        else:
            recommendation = "Send for further inspection"
    else:
        recommendation = "No action required"

    return predicted_class, confidence, recommendation

# Streamlit UI
st.title("Defect Detection in Images")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    predicted_class, confidence, recommendation = predict_defect(image)

    # Display the results
    st.write(f"Prediction: **{predicted_class}** with confidence: **{confidence:.2%}**")
    st.write(f"Recommendation: **{recommendation}**")
