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

INSTALLATION (FIRST-TIME SETUP)
-------------------------------
Step 1: Open Anaconda Prompt or Terminal and navigate to the project folder:
    cd path\to\traffic-sign-recognition-system

Step 2: Create and activate the Conda environment:
    conda create -n tsdr_env python=3.10 -y
    conda activate tsdr_env

Step 3: Install required packages:
    pip install -r requirements.txt

RUNNING THE SYSTEM
------------------
Step 1: Activate your environment (if not already active):
    conda activate tsdr_env

Step 2: Start the Flask server:
    python app.py

Step 3: Open your web browser and visit:
    http://127.0.0.1:5000

