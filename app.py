import os
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# =============================
# LOAD YOLO MODEL
# =============================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "best.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_WEIGHTS))

model = YOLO(MODEL_PATH)

# =============================
# GLOBAL STATE (MULTI DETECTION)
# =============================
current_detections = []  # list of {label, confidence}


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/how")
def how_to_use():
    return render_template("how.html")


def generate_frames():
    global current_detections

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)
            annotated = results[0].plot()

            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf >= 0.4:
                        detections.append({
                            "label": model.names[cls_id],
                            "confidence": round(conf, 2),
                        })

            current_detections = detections

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify({
        "detections": current_detections
    })


if __name__ == "__main__":
    app.run(debug=True)