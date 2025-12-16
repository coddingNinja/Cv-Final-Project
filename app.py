from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import HelperFunction
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("playingCards.pt")

classNames = [
    '10C','10D','10H','10S','2C','2D','2H','2S','3C','3D','3H','3S',
    '4C','4D','4H','4S','5C','5D','5H','5S','6C','6D','6H','6S',
    '7C','7D','7H','7S','8C','8D','8H','8S','9C','9D','9H','9S',
    'AC','AD','AH','AS','JC','JD','JH','JS','KC','KD','KH','KS',
    'QC','QD','QH','QS'
]

# ================= GLOBAL CAMERA RESULT =================
current_camera_hand = "No result yet"
current_camera_cards = []

# ================= CAMERA STREAM =================
def gen_frames():
    global current_camera_hand, current_camera_cards

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detected = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.5:
                    detected.append(classNames[cls])

        detected = list(set(detected))
        current_camera_cards = detected

        if len(detected) >= 5:
            current_camera_hand = HelperFunction.findPokerHand(detected[:5])
        else:
            current_camera_hand = "Not enough cards detected"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{classNames[cls]} {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/camera_result')
def camera_result():
    return jsonify({
        "hand": current_camera_hand,
        "cards": ", ".join(current_camera_cards)
    })

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    detected = []

    for file in files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(img)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.5:
                    detected.append(classNames[cls])

    detected = list(set(detected))
    poker_hand = "Not enough cards detected"
    if len(detected) >= 5:
        poker_hand = HelperFunction.findPokerHand(detected[:5])

    return render_template(
        "index.html",
        upload_hand=poker_hand,
        upload_cards=", ".join(detected)
    )

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
