import streamlit as st
import cv2
import numpy as np
import time
import os
import mediapipe as mp
from PIL import Image

# Judul Page
st.set_page_config(page_title="Image Classification", layout="centered")
st.title("Image Classification")

# Path Model TF Lite
model_path = os.path.join(os.getcwd(), 'model/efficientnet_lite2_float.tflite')

# Model Config
BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=3,
    running_mode=VisionRunningMode.IMAGE
)

# File Uploader
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load Image
    img_rgb = Image.open(uploaded_file).convert("RGB")
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Convert Image ke Numpy Array
    image_np = np.array(img_rgb)

    # Klasifikasi
    if st.button("🔍 Classify Image"):
        with ImageClassifier.create_from_options(options) as classifier:
            # Convert ke MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

            # Menghitung Latensi & Melakukan Inference
            start_time = time.time()
            result = classifier.classify(mp_image)
            latency = (time.time() - start_time) * 1000  # ms

        # Menampilkan Hasil Klasifikasi
        if result and result.classifications:
            st.markdown("### Top 3 Predictions:")
            for i, category in enumerate(result.classifications[0].categories):
                label = f"**{i+1}. {category.category_name}** – {category.score:.2f}"
                st.write(label)
            st.success(f"Inference Latency: {latency:.1f} ms")
        else:
            st.warning("No classification result.")