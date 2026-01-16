# Hand Gesture Volume Control 

This project allows you to control your computer's system volume using hand gestures captured from your webcam.

It uses:
- OpenCV for camera and image processing
- MediaPipe (new Tasks API) for hand detection
- PyAutoGUI to control the system volume

By changing the distance between your **thumb** and **index finger**, you can:
- Increase the volume 
- Decrease the volume 

---

## 🚀 How It Works

1. The webcam captures video in real time.
2. MediaPipe detects your hand and extracts **21 hand landmarks**.
3. The program tracks:
   - Thumb tip
   - Index finger tip
4. The distance between these two fingers is calculated.
5. If the distance:
   - Is large → Volume goes up
   - Is small → Volume goes down
6. The result is displayed on the screen with visual markers.

---

##  Requirements

Install the required Python packages:

pip install opencv-python mediapipe pyautogui numpy

## AI Model File

You must download the MediaPipe hand model:

Download from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Place it in the same folder as your Python script

## How To Run
RUN: python main.py

Press ESC to exit the program.

## Technologies Used

- Python

- OpenCV

- MediaPipe (Tasks API)

- PyAutoGUI

- NumPy

## Use Cases

- Touchless volume control

- Gesture-based human-computer interaction

- Computer vision learning project

- AI + automation demo