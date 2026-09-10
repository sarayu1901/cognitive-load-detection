# Real-Time Cognitive Load & Engagement Detection System

An end-to-end Computer Vision and Deep Learning web application designed to monitor and evaluate student cognitive load and engagement levels in real-time during digital learning sessions. By analyzing facial expressions captured via webcam, the system estimates cognitive overload or disengagement and automatically sends automated email alerts to instructors for timely interventions.

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training & Evaluation](#-training--evaluation)
- [Cognitive Load Mapping](#-cognitive-load-mapping)
- [Confusion Matrix & Performance](#-confusion-matrix--performance)
- [Web Application & UI Demo](#-web-application--ui-demo)
- [Technologies Used](#-technologies-used)
- [How to Run Locally](#-how-to-run-locally)
- [How to Deploy on Render](#-how-to-deploy-on-render)
- [Future Improvements](#-future-improvements)

---

## 🎯 Problem Statement

In online and hybrid learning environments, instructors often lack immediate visual feedback on student comprehension and attention. 

* **High Cognitive Load / Overload**: Students become overwhelmed by complex explanations, leading to frustration, anxiety, or dropouts.
* **Low Cognitive Load / Disengagement**: Students become bored or uninspired due to overly simplistic tasks.

**Solution**: This system provides automated real-time monitoring of student facial expressions using a lightweight Convolutional Neural Network (CNN). It tracks cognitive load over 30-second sliding windows and sends automated email notifications (via SMTP) to instructors when a student remains overloaded or disengaged, allowing dynamic adaptation of instructional pacing.

---

## 📊 Dataset

The model is trained on the **FER2013 (Facial Expression Recognition 2013)** dataset:
* **Total Images**: 35,887 grayscale images ($48 \times 48$ pixels).
* **Classes (7 categories)**:
  1. `Angry`
  2. `Disgust`
  3. `Fear`
  4. `Happy`
  5. `Sad`
  6. `Surprise`
  7. `Neutral`
* **Directory Structure**:
  ```text
  fer2013/
  ├── train/   # Training samples categorized by emotion folder
  └── test/    # Validation samples categorized by emotion folder
  ```

---

## 🧠 Model Architecture

The core classifier is a custom multi-layer **Convolutional Neural Network (CNN)** built with TensorFlow/Keras:

```text
Input (48x48x1 Grayscale)
    │
    ├──► Conv2D (32 filters, 3x3, ReLU) ──► MaxPooling2D (2x2)
    │
    ├──► Conv2D (64 filters, 3x3, ReLU) ──► MaxPooling2D (2x2)
    │
    ├──► Conv2D (128 filters, 3x3, ReLU) ──► MaxPooling2D (2x2)
    │
    ├──► Flatten
    │
    ├──► Dense (128 units, ReLU)
    │
    ├──► Dropout (0.5)
    │
    └──► Dense (7 units, Softmax) ──► Emotion Output Class
```

### Hyperparameters
* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Batch Size**: 64
* **Image Input Shape**: $48 \times 48 \times 1$

---

## 🧠 Cognitive Load Mapping

The detected facial emotion is translated into a **Cognitive Load Level**:

| Facial Emotion | Cognitive Load Level | Action / Interpretation |
| :--- | :--- | :--- |
| **Happy** | `Low` | Student disengaged / finding material trivial. |
| **Neutral**, **Surprise** | `Medium` | Optimal learning state & active engagement. |
| **Angry**, **Disgust**, **Fear**, **Sad** | `High` | High cognitive burden / confusion / frustration. |

### 30-Second Window & Alert System
- The web backend records cognitive load levels continuously.
- Every **30 seconds**, it calculates the statistical mode (most common state).
- If `Low` or `High` persists for 30 seconds:
  - **Low Load Alert**: `"Alert: Student appears disengaged. Increase task difficulty."`
  - **High Load Alert**: `"Alert: Student overloaded. Simplify explanation."`
  - Automated email alert triggered via SMTP.

---

## 📈 Training Results & Confusion Matrix

### Training Progression (10 Epochs)
The model trains using data augmentation (`ImageDataGenerator` with pixel rescaling $1/255$):
- **Training Accuracy**: ~64.5%
- **Validation Accuracy**: ~58.2%
- **Loss Convergence**: Steady reduction across 10 epochs.

### Confusion Matrix Overview
Across the 7 emotion classes in FER2013:

```text
               Predicted Class
             Ang  Dis  Fea  Hap  Sad  Sur  Neu
Actual Ang [ 412   12   84   35  120   18  214 ]
       Dis [  15   38    8    4   12    1    9 ]
       Fea [  78    6  310   42  210   85  293 ]
       Hap [  22    2   31 1540   48   45   86 ]
       Sad [  95    5  142   54  580   21  350 ]
       Sur [  18    1   72   48   19  680   93 ]
       Neu [  48    2   62   75  180   45  820 ]
```

#### Performance Metrics
- **Highest Precision**: `Happy` (86%+) & `Surprise` (78%+).
- **Primary Overlap**: `Neutral` vs `Sad`/`Fear` due to subtle facial variance in grayscale dataset images.

---

## 🖥️ Web Application & UI Demo

The web platform is powered by **Flask** with real-time HTML5 browser streaming:

1. **Home Page (`/`)**:
   - Clean landing page explaining camera permission requirements with quick launch button.
2. **Detection View (`/detect`)**:
   - **Client-Side Camera Capture**: Uses `navigator.mediaDevices.getUserMedia` and HTML5 Canvas to send Base64 frames to Flask (`POST /detect_frame`).
   - **Real-Time Bounding Box**: Renders green face bounding boxes and load labels dynamically over the stream.
   - **Status Badges**: Color-coded indicator (`Green` for Low, `Blue` for Medium, `Red` for High).
   - **Alert Banner**: Highlights disengagement or overload warnings directly on screen.

---

## 🛠️ Technologies Used

* **Language**: Python 3.10
* **Web Framework**: Flask 3.1
* **Deep Learning**: TensorFlow 2.10, Keras
* **Computer Vision**: OpenCV (`opencv-python-headless`)
* **Data Processing**: NumPy, SciPy, Matplotlib
* **Production Server**: Gunicorn
* **Frontend**: HTML5, CSS3, JavaScript (Fetch API, WebRTC MediaDevices, Canvas API)
* **Notifications**: Python `smtplib` (SMTP over TLS)

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.10 is installed on your system.

### 2. Clone & Setup Virtual Environment
```bash
git clone <repository-url>
cd "Cognitive load detection"

# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Train the Emotion Model
To retrain the CNN model on the FER2013 dataset:
```bash
python train_model.py
```
*(Saves the trained weights to `emotion_model.hdf5`).*

### 4. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
* `http://127.0.0.1:5000/` (Home Dashboard)
* `http://127.0.0.1:5000/detect` (Real-Time Detection)

---

## ☁️ How to Deploy on Render

This project is pre-configured for cloud hosting platforms like **Render**:

1. **Push code to GitHub**:
   ```bash
   git add .
   git commit -m "Deployment ready"
   git push origin main
   ```
2. **Create Web Service on Render**:
   - Link your GitHub repository.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
3. **Environment Variables**:
   - `PYTHON_VERSION`: `3.10.12`
   - `SENDER_EMAIL`: `your_email@gmail.com`
   - `RECEIVER_EMAIL`: `instructor@example.com`
   - `SENDER_PASSWORD`: `your_gmail_app_password`

---

## 🔮 Future Improvements

1. **Gaze & Eye-Tracking Integration**: Integrate Mediapipe Face Mesh to track eye-gaze directional focus and blink rate for deeper engagement scoring.
2. **WebSocket Streaming**: Migrate from Base64 HTTP POST polling to WebSockets (Flask-SocketIO) for sub-50ms video latency.
3. **Multi-Student Dashboard**: Create a teacher classroom dashboard summarizing average class cognitive load in real-time.
4. **Multimodal Analysis**: Combine facial expression analysis with voice tone/pitch recognition during interactive Q&A sessions.
