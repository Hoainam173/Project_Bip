import cv2
import cvzone
import math
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

# Function to process and detect falls in a frame
def detect_fall(frame, prev_time_ref, fall_timer_ref):
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
    fps = 1 / (curr_time - prev_time_ref[0]) if prev_time_ref[0] else 0
    prev_time_ref[0] = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if fall_detected:
        if time.time() - fall_timer_ref[0] > 5:
            pygame.mixer.Sound.play(alert_sound)
            fall_timer_ref[0] = time.time()

    return frame, fall_detected

def main():
    st.title("Real-Time Fall Detection")
    st.sidebar.header("Control Panel")
    st.sidebar.text("This application detects falls in real-time using YOLOv8.")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    prev_time_ref = [0]
    fall_timer_ref = [0]

    video_placeholder = st.empty()
    warning_placeholder = st.empty()
    status_placeholder = st.empty()

    frame_skip = 1
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected or disconnected.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            processed_frame, fall_detected = detect_fall(frame, prev_time_ref, fall_timer_ref)

            img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_rgb = np.array(img_rgb)

            video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

            if fall_detected:
                warning_placeholder.warning("FALL DETECTED")
                status_placeholder.text("Warning: Fall detected!")
            else:
                warning_placeholder.empty()
                status_placeholder.text("Monitoring...")

            time.sleep(0.05)  # Control frame rate

    except st.errors.StreamlitAPIException:
        # Occurs when the user stops the Streamlit app
        pass
    finally:
        cap.release()

if __name__ == "__main__":
    main()

