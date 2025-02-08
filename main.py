import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile

# Load your deepfake detection model
model = tf.keras.models.load_model("deepfake-detection-model1.h5")

st.title("DeepShield \n Deepfake Video Detection")

# Video Upload
uploaded_video = st.file_uploader("Upload a video to check for deepfake", type=["mp4", "mov", "avi"])

if uploaded_video:
    # Save the video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    st.video(temp_file.name)

    # Extract frames using OpenCV
    cap = cv2.VideoCapture(temp_file.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predictions = []

    st.write("Processing video frames...")
    with st.spinner("KIRMADA is detecting deepfakes..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for the model
            resized_frame = cv2.resize(frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            # Predict frame
            prediction = model.predict(input_frame)[0][0]
            predictions.append(prediction)

        cap.release()

    # Calculate the average deepfake score
    average_score = np.mean(predictions)
    deepfake_percentage = average_score * 100

    # Display results
    st.subheader("Deepfake Detection Results")
    st.write(f"Likelihood of Deepfake: {deepfake_percentage:.2f}%")
    
    if deepfake_percentage > 50:
        st.error("The video is likely a DEEPFAKE.")
    else:
        st.success("The video is likely AUTHENTIC.")
