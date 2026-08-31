import cv2
import numpy as np
import time
import smtplib
from collections import Counter
from flask import Flask, render_template, Response, jsonify
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

    sender_email = "saicharanuppala01@gmail.com"
    receiver_email = "dindukurthi.sarayu2024@vitstudent.ac.in"
    password = "apyw wizq uaxy issk"

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
        print("Email sent")

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


# ---------------- VIDEO PROCESSING ----------------
def generate_frames():

    global current_load, load_history, start_time, alert_message

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (64,64))

            face = face / 255.0

            face = np.reshape(face,(1,64,64,1))

            prediction = model.predict(face,verbose=0)

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

            cv2.putText(frame, f"Load: {current_load}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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


@app.route('/load')
def load():

    return jsonify({"load": current_load})


@app.route('/alert')
def alert():

    return jsonify({"alert": alert_message})


if __name__ == "__main__":

    app.run(debug=True)