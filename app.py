%%writefile app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import base64

# NEW SETUP (static + ngrok compatible)
app = Flask(__name__, static_folder="static")

emotion_labels = [
    'angry','disgust','fear',
    'happy','neutral','sad','surprise'
]

model = tf.keras.models.load_model("Face_Recognation_model1.keras")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # base64 ? image
    img_data = base64.b64decode(data.split(",")[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #  OLD BEHAVIOUR RESTORED (less strict, smooth)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # old feel (better realtime)
        minNeighbors=4,      # less strict
        minSize=(60, 60)     # door face bhi detect
    )

    # OLD LOGIC STYLE (no hard break)
    if len(faces) == 0:
        return jsonify({
            "status": "no_face",
            "message": "Face not detected"
        })

    #  ALWAYS USE FIRST FACE (old smooth behaviour)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (128,128))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    preds = model.predict(face, verbose=0)[0]
    idx = np.argmax(preds)

    return jsonify({
        "status": "ok",
        "emotion": emotion_labels[idx],
        "confidence": float(preds[idx] * 100)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
