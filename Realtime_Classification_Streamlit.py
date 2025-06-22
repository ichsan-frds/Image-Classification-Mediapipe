import streamlit as st
import cv2
import numpy as np
import time
import os
import mediapipe as mp

# Judul Page
st.set_page_config(page_title="🧠 Real-Time Classifier", layout="centered")
st.title("Real-Time Image Classification with MediaPipe")

# State Management
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Path Model TF Lite
model_path = os.path.join(os.getcwd(), 'model/efficientnet_lite2_float.tflite')

# Model Config
BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=1,
    running_mode=VisionRunningMode.IMAGE
)

# Function Toggle State
def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on

# Dynamic Button (dipengaruhi State)
button_label = "🟢 Start Camera" if not st.session_state.camera_on else "🔴 Stop Camera"
st.button(button_label, on_click=toggle_camera)

# Placeholder untuk Output Webcam
frame_placeholder = st.empty()
label_placeholder = st.empty()

# Selama Webcam menyala, lakukan klasifikasi
if st.session_state.camera_on:
    with ImageClassifier.create_from_options(options) as classifier:
        # Mengambil gambar dari Webcam
        cap = cv2.VideoCapture(0)

        # Error Handling Webcam tidak accessible
        if not cap.isOpened():
            st.error("Camera not accessible.")
            st.session_state.camera_on = False
        else:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame capture failed.")
                    break

                # Convert ke RGB lalu Mediapipe Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Menghitung Latensi & Melakukan Inference
                start_time = time.time()
                result = classifier.classify(mp_image)
                latency = (time.time() - start_time) * 1000  # ms

                # Menyimpan hasil klasifikasi
                annotated_img = frame.copy()
                label_text = "No result"
                if result and result.classifications:
                    top_class = result.classifications[0].categories[0]
                    label_text = f"{top_class.category_name} ({top_class.score:.2f})"

                # Menampilkan hasil klasifikasi
                frame_placeholder.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB")
                label_placeholder.markdown(f"**Prediction:** {label_text}  |  **Latency:** {latency:.1f} ms")

                time.sleep(0.05)

            cap.release()
            frame_placeholder.empty()
            label_placeholder.empty()