import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import json

# Load the pre-trained model
try:
    model = tf.keras.models.load_model("model.h5")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.h5' is in the correct directory.")
    st.stop()

# Define label mapping for traffic sign detection
label_map = {
    0: "Stop Sign",
    1: "Yield Sign",
    2: "Speed Limit Sign",
    3: "Pedestrian Crossing Sign",
    4: "No Entry Sign"
}


def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Resize image to match the model's input shape
    return np.array(img) / 255.0  # Normalize the image


def load_lottie_file(path: str):
    with open(path, "r") as f:
        return json.load(f)


st.set_page_config(page_title="🚦 Traffic Sign Detection", layout="wide")

# Page background color
page_bg_color = {
    "Home": "#E6F4F1",
    "Sign Detection": "#E6F4F1",
    "More Information": "#E6F4F1"
}

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

with st.container():
    st.markdown(
        f"""
        <style>
        .block-container {{
            background-color: {page_bg_color[st.session_state.current_page]};
        }}
        .horizontal-buttons > div {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }}
        </style>
        """, unsafe_allow_html=True
    )

    cols = st.columns(3)
    nav_labels = ["Home", "Sign Detection", "More Information"]
    for i, label in enumerate(nav_labels):
        if cols[i].button(label):
            st.session_state.current_page = label

if st.session_state.current_page == "Home":
    st.title("🚦 Traffic Sign Detection")
    st.subheader("👋 Welcome to the Traffic Sign Recognition App!")

    # Load animation with transparent background
    lottie_animation = load_lottie_file("Animation - TrafficSigns.json")

    # Centered animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st_lottie(lottie_animation, height=300, key="traffic", quality="high", speed=1)

    st.markdown("""
        This app uses deep learning to recognize traffic signs in images:
        - Upload traffic sign images 🖼️
        - Get real-time recognition and instructions 🚦

        🚨 Disclaimer: This app is intended for educational purposes. Please use responsibly!
    """)

elif st.session_state.current_page == "Sign Detection":
    st.header("🖼️ Upload Image for Traffic Sign Recognition")

    image_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = load_image(image_file)
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        image_array = np.expand_dims(image, axis=0)
        prediction = model.predict(image_array)
        label = np.argmax(prediction)

        st.success(f"✅ Prediction: {label_map[label]}")
        advice_map = {
            "Stop Sign": "🛑 Please halt immediately and wait until the road is clear.",
            "Yield Sign": "⚠️ Slow down and give way to other vehicles or pedestrians.",
            "Speed Limit Sign": "🚗 Adjust your speed to comply with the posted limit.",
            "Pedestrian Crossing Sign": "🚶 Be cautious and yield to pedestrians crossing the road.",
            "No Entry Sign": "❌ Do not enter this lane or road. It is prohibited."
        }
        st.markdown(f"*🧭 Advice:* {advice_map[label_map[label]]}")

elif st.session_state.current_page == "More Information":
    st.header("📘 About This App")

    st.markdown("""
        <div style='line-height:1.8; font-size:18px;'>
        <ul>
            <li>🤖 This app uses deep learning to identify traffic signs.</li>
            <li>🖼️ Upload a traffic sign image, and the app will recognize it.</li>
            <li>🔍 The AI model is trained on traffic sign datasets to classify signs into different categories.</li>
            <li>🚦 Based on the detected sign, it provides instructions to ensure safe driving.</li>
            <li>⚠️ <b>Disclaimer:</b> This tool is for informational and educational purposes only. It is <u>not a replacement</u> for real-time traffic systems.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)