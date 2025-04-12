import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Load EZ-HOI-inspired VLM model and processor [[1]]
@st.cache_resource
def load_model():
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")  # Replace with EZ-HOI or similar
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")  # Replace with EZ-HOI or similar
    return model, processor

model, processor = load_model()

# Define text prompts for zero-shot classification [[6]]
prompts = [
    "a worker wearing a safety helmet and reflective coat",  # Compliant
    "a worker without safety gear"                          # Non-compliant
]

# Initialize face detection model [[7]]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process a single image
def process_image(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    non_compliant_count = 0  # Counter for non-compliant workers

    for (x, y, w, h) in faces:
        face_img = img_rgb[y:y+h, x:x+w]  # Crop face region
        face_pil = Image.fromarray(face_img)

        # Prepare inputs for VLM with guided prompts [[1]]
        inputs = processor(images=face_pil, text=prompts, return_tensors="pt", padding=True)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Compute probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()[0]
        compliance_status = "Compliant" if probs[0] > probs[1] else "Non-Compliant"

        # Update non-compliant count
        if compliance_status == "Non-Compliant":
            non_compliant_count += 1

        # Annotate face with compliance status
        color = (0, 255, 0) if compliance_status == "Compliant" else (0, 0, 255)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_rgb, compliance_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_rgb, non_compliant_count

# Function to process a video
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    stframe = st.empty()  # Placeholder for video frames
    non_compliant_count = 0  # Counter for non-compliant workers

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]  # Crop face region
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Prepare inputs for VLM with guided prompts [[1]]
            inputs = processor(images=face_pil, text=prompts, return_tensors="pt", padding=True)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Compute probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).tolist()[0]
            compliance_status = "Compliant" if probs[0] > probs[1] else "Non-Compliant"

            # Update non-compliant count
            if compliance_status == "Non-Compliant":
                non_compliant_count += 1

            # Annotate face with compliance status
            color = (0, 255, 0) if compliance_status == "Compliant" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, compliance_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display annotated frame
        stframe.image(frame, channels="BGR")

    cap.release()
    return non_compliant_count

# Streamlit UI and Dashboard
st.title("Refinery PPE Compliance Dashboard")

# Sidebar for metrics
st.sidebar.header("Compliance Metrics")
non_compliant_metric = st.sidebar.empty()  # Placeholder for total non-compliant count

# Option to upload multiple photos
uploaded_photos = st.file_uploader("Upload a group of photos for PPE scanning", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_photos:
    total_non_compliant_count = 0  # Total counter for all images

    st.subheader("Processing Uploaded Photos...")
    for i, uploaded_photo in enumerate(uploaded_photos):
        st.write(f"Processing Image {i+1}: {uploaded_photo.name}")
        image = Image.open(uploaded_photo)

        # Process the image
        annotated_image, non_compliant_count = process_image(image)
        total_non_compliant_count += non_compliant_count

        # Display the annotated image
        st.image(annotated_image, caption=f"Processed Image {i+1}", use_column_width=True)

    # Update dashboard metrics
    non_compliant_metric.metric(label="Total Non-Compliant Workers", value=total_non_compliant_count)

# Option to upload a video file
uploaded_video = st.file_uploader("Upload a video for PPE scanning", type=["mp4", "avi"])

if uploaded_video:
    # Save uploaded file temporarily
    temp_file = f"temp_video.{uploaded_video.name.split('.')[-1]}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_video.read())

    # Process video and get non-compliant count
    st.subheader("Processing Uploaded Video...")
    non_compliant_count = process_video(temp_file)

    # Update dashboard metrics
    non_compliant_metric.metric(label="Non-Compliant Workers in Video", value=non_compliant_count)