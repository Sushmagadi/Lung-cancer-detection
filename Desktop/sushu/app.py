import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit Page Configuration must be the first command
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    """Loads the pre-trained VGG16 model for lung cancer detection."""
    return tf.keras.models.load_model("Lung_Cancer_Detection_using_Vgg16.keras")

# Define class names (must match your dataset)
CLASS_NAMES = ['Lung Adenocarcinoma', 'Normal', 'Lung Squamous Cell Carcinoma']

# Preprocessing function
def preprocess_image(img):
    """
    Preprocesses the uploaded image for the VGG16 model.
    Converts to RGB, resizes, and normalizes the pixel values.
    """
    img = img.convert("RGB")
    
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Load the model outside the main app loop
model = load_model()

st.title("🫁 Lung Cancer Detection using VGG16")
st.write("Upload a lung scan image to detect whether it is **Normal**, **Adenocarcinoma**, or **Squamous Cell Carcinoma**.")
st.markdown("---")

# File uploader widget
uploaded_file = st.file_uploader("📤 Upload Lung Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Use a spinner while the model is predicting
    with st.spinner("Analyzing the image..."):
        # Preprocess the image and get the prediction
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)

    # Get the predicted class and confidence score
    score = np.max(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.markdown("---")
    st.subheader("🔎 Prediction Result")

    # Display the result with an appropriate icon and color
    if predicted_class == "Normal":
        st.success(f"✅ Predicted Class: **{predicted_class}**")
        st.balloons()
    else:
        st.error(f"⚠️ Predicted Class: **{predicted_class}**")
        st.warning("This analysis is for informational purposes only. Please consult a medical professional for a definitive diagnosis.")

    st.info(f"Confidence: {score * 100:.2f}%")

    # Probabilities chart
    st.subheader("📊 Class Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]  # Blue, Green, Red
    
    # Check for the case of a single bar chart
    if len(prediction[0]) == len(CLASS_NAMES):
        ax.bar(CLASS_NAMES, prediction[0], color=colors)
    else:
        # Fallback for unexpected shapes
        ax.bar(range(len(prediction[0])), prediction[0], color=colors)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")

    ax.set_ylabel("Probability")
    ax.set_xlabel("Classes")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)