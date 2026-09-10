import os
import cv2
import numpy as np
import time
import smtplib
import base64
from collections import Counter
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
from email.mime.text import MIMEText

app = Flask(__name__)

model = load_model("emotion_model.hdf5", compile=False)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

current_load = "Detecting..."
load_history = []
start_time = time.time()
alert_message = ""

# ---------------- EMAIL FUNCTION ----------------
def send_alert_email(message):

    sender_email = os.environ.get("SENDER_EMAIL", "saicharanuppala01@gmail.com")
    receiver_email = os.environ.get("RECEIVER_EMAIL", "dindukurthi.sarayu2024@vitstudent.ac.in")
    password = os.environ.get("SENDER_PASSWORD", "apyw wizq uaxy issk")

    msg = MIMEText(message)
    msg['Subject'] = "Cognitive Load Alert"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")

    except Exception as e:
        print("Email error:", e)

# ---------------- LOAD MAPPING ----------------
def get_cognitive_load(emotion):

    if emotion == "Happy":
        return "Low"

    elif emotion in ["Neutral", "Surprise"]:
        return "Medium"

    else:
        return "High"

# ---------------- FRAME PROCESSING ----------------
def process_frame_data(frame):
    global current_load, load_history, start_time, alert_message

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = model.predict(face, verbose=0)
        emotion = emotions[np.argmax(prediction)]

        current_load = get_cognitive_load(emotion)
        load_history.append(current_load)

        # -------- 30 SECOND WINDOW --------
        if time.time() - start_time >= 30:
            if load_history:
                avg_load = Counter(load_history).most_common(1)[0][0]

                if avg_load == "Low":
                    alert_message = "Student disengaged for 30 seconds."
                    send_alert_email("Alert: Student appears disengaged. Increase task difficulty.")
                elif avg_load == "Medium":
                    alert_message = "Student learning normally."
                else:
                    alert_message = "Student overloaded for 30 seconds."
                    send_alert_email("Alert: Student overloaded. Simplify explanation.")

            load_history.clear()
            start_time = time.time()

        cv2.putText(frame, f"Load: {current_load}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# ---------------- VIDEO PROCESSING (LOCAL HARDWARE FALLBACK) ----------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame_data(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("home.html")


@app.route('/detect')
def detect():
    return render_template("detect.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image frame"}), 400

        annotated_frame = process_frame_data(frame)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            encoded_annotated = base64.b64encode(buffer).decode('utf-8')
            annotated_url = f"data:image/jpeg;base64,{encoded_annotated}"
        else:
            annotated_url = ""

        return jsonify({
            "load": current_load,
            "alert": alert_message,
            "image": annotated_url
        })
    except Exception as e:
        print("Frame processing error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/load')
def load():
    return jsonify({"load": current_load})


@app.route('/alert')
def alert():
    return jsonify({"alert": alert_message})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)