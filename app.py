import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

emotion_labels = [
    'angry','disgust','fear',
    'happy','neutral','sad','surprise'
]

model = tf.keras.models.load_model("Face_Recognation_model1.keras")

st.set_page_config(page_title="Face Emotion Detection", layout="centered")
st.title("Face Emotion Detection")

img_file = st.camera_input("Capture your face")

if img_file is not None:
    image = Image.open(img_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning("Face not detected")
    else:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (128,128))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        idx = np.argmax(preds)

        st.success(f"Emotion: {emotion_labels[idx]}")
        st.info(f"Confidence: {preds[idx]*100:.2f}%")
