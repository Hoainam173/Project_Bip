# Project_BioimageProcessing
I. Real-Time Fall Detection System
This project is a real-time fall detection application using the YOLOv8 object detection model, OpenCV, and a Tkinter-based GUI. It aims to detect when a person falls using a webcam and trigger a warning alert.

II. Features
- Real-time detection with YOLOv8 (yolov8n) for speed and efficiency
- Alerts with visual and audio warnings when a fall is detected
- Tkinter GUI for simple monitoring
- Displays FPS and detection boxes for feedback
- Runs on GPU if available for better performance

III. Requirements
- Make sure to install the following dependencies:
pip install ultralytics opencv-python cvzone pillow pygame torch torchvision torchaudio
- You also need to download:
The YOLOv8 model weights: yolov8n.pt (you can get it from Ultralytics)
An alert sound file: alert.wav (place it in the same directory)

IV. How to Run
1. Ensure your webcam is connected.
2. Run the script:
python fall_detection.py
3. The GUI will pop up, and the webcam feed will start.
4. When a fall is detected:
- A red box and a "FALL DETECTED" message appear
- An alert sound is played
- Status updates in the GUI

V. How It Works
- The YOLO model detects objects in the frame.
- For each detected person, it calculates the bounding box dimensions.
- If a box is wider than it is tall, it's classified as a potential fall.
- A sound is triggered and a visual alert is shown if a fall is detected and not triggered in the last 5 seconds.

VI. Screenshot
(Optional: Add a screenshot of the app GUI with a detected fall here.)

***Notes**
* The model is set to detect only the person class with confidence > 80%.
* You can improve accuracy by training a custom YOLO model on fall detection datasets.
* For low-end systems, reducing frame size or skipping more frames may help.

***License**
MIT License – free to use and modify.


