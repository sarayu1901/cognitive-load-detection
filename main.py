# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion model
model = load_model("emotion_model.hdf5")

# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to convert emotion to cognitive load
def get_cognitive_load(emotion):
    if emotion in ["Happy"]:
        return "Low"
    elif emotion in ["Neutral", "Surprise"]:
        return "Medium"
    else:
        return "High"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64,64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = model.predict(face)
        emotion = emotions[np.argmax(prediction)]

        load = get_cognitive_load(emotion)

        cv2.putText(frame, f"Cognitive Load: {load}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y),
                      (x+w, y+h),
                      (0, 255, 0), 2)

    cv2.imshow("Cognitive Load Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
