import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_and_annotate(image):
    """Process the image and annotate it with pose landmarks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    annotated_image = image.copy()

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return annotated_image

def main():
    st.set_page_config(page_title="Pose Estimation Hub", page_icon="🤸", layout="wide")

    # Header Section
    st.markdown("""
    <style>
    .header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
    }
    </style>
    <div class="header">🤸 Pose Estimation Hub</div>
    <div class="subheader">Analyze human poses with advanced AI-based techniques</div>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.sidebar.header("Upload Your Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_image = process_and_annotate(image_bgr)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.write("### Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Pose Annotated Image", use_column_width=True)

        st.success("Pose analysis completed! View the annotated image on the right.")
    else:
        st.info("Upload an image from the sidebar to begin pose estimation.")

    # Additional Sections
    st.markdown("""
    ---
    ### About Pose Estimation
    Pose estimation is a technique to detect human body keypoints, like shoulders, elbows, and knees, in images or videos. 
    It has applications in:
    - Fitness Tracking
    - Healthcare & Rehabilitation
    - Sports Analytics
    - Augmented Reality

    ---
    ### How It Works
    1. Upload an image in the sidebar.
    2. Our model processes it using [Mediapipe](https://mediapipe.dev/).
    3. The annotated image is displayed alongside the original.
    
    Developed with ❤️ by Shiva.
    """)

if __name__ == "__main__":
    main()

pose.close()
