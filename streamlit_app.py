import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO
from HelperFunction import findPokerHand

st.title("🃏 Poker Hand Detection")

model = YOLO("playingCards.pt")

classNames = ['10C','10D','10H','10S','2C','2D','2H','2S','3C','3D','3H','3S',
              '4C','4D','4H','4S','5C','5D','5H','5S','6C','6D','6H','6S',
              '7C','7D','7H','7S','8C','8D','8H','8S','9C','9D','9H','9S',
              'AC','AD','AH','AS','JC','JD','JH','JS','KC','KD','KH','KS',
              'QC','QD','QH','QS']

uploaded = st.file_uploader("Upload poker card image", type=["jpg","png","jpeg"])

if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model(img, stream=True)
    hand = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.5:
                hand.append(classNames[cls])

    hand = list(set(hand))
    result = findPokerHand(hand)

    st.image(img, channels="BGR")
    st.success(f"Detected Hand: {result}")
