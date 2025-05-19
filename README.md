# 🚀 Real-Time Fall Detection System using YOLOv8 and Camera

This project is a **real-time fall detection system** that utilizes the **YOLOv8 object detection model** and **camera input** to monitor and detect falls. It is designed for applications in health monitoring, elderly care, and workplace safety, providing immediate alerts whenever a fall is detected.

---

## ⚡ Features:

✅ **Real-Time Detection:** Uses YOLOv8 (yolov8n) for fast and efficient object detection, even on standard hardware.
✅ **Accurate Fall Identification:** Detects falls by analyzing the aspect ratio of detected person’s bounding box.
✅ **Instant Alerts:** Visual alerts on the screen and an audio warning sound (alert.wav) are triggered when a fall is detected.
✅ **Adaptive Performance:** Automatically utilizes GPU (if available) for faster processing.
✅ **Configurable Thresholds:** Adjustable confidence level for person detection (default: >80%).
✅ **FPS Display:** Real-time frame rate monitoring for performance tracking.

---

## 📦 Requirements:

1. Install dependencies using:

```bash
pip install ultralytics opencv-python cvzone pillow pygame torch torchvision torchaudio streamlit
```

2. Ensure you have:

   * **YOLOv8 model weights:** Download `yolov8n.pt` from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
   * **Alert Sound File:** An `alert.wav` file placed in the same directory.

---

## 🖥️ How to Run:

1. Connect your webcam.
2. Run the application using Streamlit:

```bash
streamlit run fall_detection.py
```

3. The camera feed will appear in the browser interface.
4. When a fall is detected:

   * A red bounding box appears around the detected person.
   * A "FALL DETECTED" message is displayed.
   * An alert sound is triggered.

---

## 🧠 How It Works:

* The YOLOv8 model detects objects in the camera feed, focusing on the "person" class.
* For each detected person, the bounding box dimensions are analyzed:

  * If the width of the box is greater than its height, it is classified as a potential fall.
* If a fall is detected, a sound alert is played, and a warning message is displayed on the screen.
* A 5-second delay prevents repeated alerts for the same fall.

---

## 📌 Optimization Tips:

* If running on a low-end system, consider reducing frame size or skipping frames to improve performance.
* For better accuracy, consider training a custom YOLO model on a fall detection dataset.
* Adjust the detection confidence threshold to match your environment.

---

## 📄 License:

MIT License – Free to use, modify, and distribute.
