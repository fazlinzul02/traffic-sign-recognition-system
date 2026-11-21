# Real-Time Traffic Sign Detection & Recognition System (TSDR)

A real-time web-based Traffic Sign Detection and Recognition (TSDR) system powered by **Ultralytics YOLO**, **OpenCV**, and **Flask**. The application captures live camera frames, runs inference using custom-trained YOLO weights (`best.pt`), and streams annotated video alongside live detection results and confidence scores to a web interface.

---

## Key Features

- **Real-Time Multi-Object Detection:** Processes live camera frames via OpenCV (`cv2.VideoCapture`) and detects multiple traffic signs simultaneously with bounding boxes and class labels.
- **Confidence Filtering:** Filters out low-confidence predictions (`confidence >= 0.4`) to reduce false positives during live streaming.
- **Live MJPEG Video Streaming:** Streams annotated frames smoothly to the browser over HTTP (`/video_feed`).
- **Real-Time Detection Status API:** Exposes a `/status` JSON endpoint returning detected class names and rounded confidence scores for dynamic frontend updates.
- **Multi-Page Web Interface:** Includes dedicated pages for system introduction (`/`), live detection (`/live`), and usage instructions (`/how`).

---

## Tech Stack

- **Backend & Routing:** Python, Flask
- **Deep Learning & Computer Vision:** Ultralytics YOLO, OpenCV (`opencv-python`)
- **Frontend:** HTML (`welcome.html`, `live.html`, `how.html`)

---

## Project Structure

```text
traffic-sign-recognition-system/
├── app.py                  # Flask server, YOLO inference pipeline, and API routes
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── weights/
│   └── best.pt             # Custom-trained YOLO model weights
└── templates/
    ├── welcome.html        # Landing / welcome page
    ├── live.html           # Live camera detection interface
    └── how.html            # User guide / how-to-use page

# How to Use: Traffic Sign Detection & Recognition System (TSDR)

This guide explains how to set up, run, and use the TSDR web application with your camera feed.

---

## Before You Start

Make sure you have:
- A working camera connected to your device (default camera index `0`).
- No other application (such as Zoom, Teams, or another camera app) currently locking the camera.
- Your environment (`tsdr_env`) set up with dependencies installed from `requirements.txt`.
- Your trained model weights placed at `weights/best.pt`.

---

## Step-by-Step Guide

### 1. Activate Your Environment and Start the Server
Open your terminal or Anaconda Prompt inside the project folder (`traffic-sign-recognition-system`), activate your environment, and run `app.py`:

```bash
conda activate tsdr_env
python app.py
