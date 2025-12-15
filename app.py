from flask import Flask, request, render_template
import numpy as np
from ultralytics import YOLO
from PIL import Image
from HelperFunction import findPokerHand
import os

app = Flask(__name__)
model = YOLO("playingCards.pt")

classNames = ['10C','10D','10H','10S','2C','2D','2H','2S','3C','3D','3H','3S',
              '4C','4D','4H','4S','5C','5D','5H','5S','6C','6D','6H','6S',
              '7C','7D','7H','7S','8C','8D','8H','8S','9C','9D','9H','9S',
              'AC','AD','AH','AS','JC','JD','JH','JS','KC','KD','KH','KS',
              'QC','QD','QH','QS']

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).convert("RGB")
        img_np = np.array(image)

        detections = model(img_np)
        hand = []

        for r in detections:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf > 0.5:
                    hand.append(classNames[cls])

        hand = list(set(hand))
        result = findPokerHand(hand)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()
