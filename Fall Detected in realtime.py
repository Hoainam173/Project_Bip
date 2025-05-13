import cv2
import cvzone
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import pygame
import torch
import streamlit as st
import numpy as np
from ultralytics import YOLO

# Initialize pygame to play the alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Load YOLO model (use yolov8n.pt for better real-time performance)
model = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    model.to('cuda')
classnames = model.names

# Use webcam (0 = default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Streamlit app title and sidebar
st.title("Real-Time Fall Detection")
st.sidebar.header("Control Panel")
st.sidebar.text("This application detects falls in real-time using YOLOv8.")

# Initialize variables
prev_time = 0
fall_timer = 0
frame_skip = 1
frame_count = 0

# Create a placeholder for the video feed
video_placeholder = st.empty()

# Start the video capture and processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not detected or disconnected.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, verbose=False)

    fall_detected = False

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            class_name = classnames.get(cls_id, f"Unknown({cls_id})")
            confidence = math.ceil(conf * 100)

            if class_name != 'person' or confidence < 80:
                continue

            width = x2 - x1
            height = y2 - y1
            is_fall = width > height

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6, colorC=color)
            cvzone.putTextRect(frame, f'{class_name} {confidence}%', [x1 + 8, y1 - 12], thickness=2, scale=1)

            if is_fall:
                fall_detected = True
                cvzone.putTextRect(frame, ' FALL DETECTED ', [x1, y2 + 20], scale=2, thickness=2, colorR=(0, 0, 255))

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    if fall_detected:
        if time.time() - fall_timer > 5:
            pygame.mixer.Sound.play(alert_sound)
            fall_timer = time.time()

    # Convert frame to RGB for Streamlit
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = np.array(img_rgb)

    # Update Streamlit display
    video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)
    if fall_detected:
        st.warning("FALL DETECTED")
    else:
        st.success("Monitoring...")

    # Add a small delay to control the frame rate
    time.sleep(0.1)

# Release the video capture when done
cap.release()
