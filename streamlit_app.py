import streamlit as st
import numpy as np
import math
from ultralytics import YOLO
from PIL import Image
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
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    results = model(img_np)
    hand = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.5:
                hand.append(classNames[cls])

    hand = list(set(hand))
    result = findPokerHand(hand)

    st.image(image, caption="Uploaded Image")
    st.success(f"Detected Hand: {result}")
