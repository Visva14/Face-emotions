<h1 align="center">🎭 Face Emotion Detection System</h1>

<p align="center">
A deep learning project that teaches machines to read human emotions — from static images to real-time webcam feeds.
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:00C6FF,100:7A00FF&height=4" width="100%">
</p>

---

## 🧠 Project Overview

This project focuses on **facial emotion recognition** using **Convolutional Neural Networks (CNNs)**.  
The system is trained on labeled facial images and is capable of predicting human emotions both from datasets and **live webcam input**.

The goal of this project is to explore how deep learning models interpret facial expressions and translate subtle visual patterns into meaningful emotional categories.

---

## 😊 Emotions Detected

The model classifies faces into **7 core emotions**:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

Each emotion is treated as a separate class during training and prediction.

---

## 📁 Project Structure (Explained)
```bash
Face emotions/
│
├── Face_reactions.py
│ ├─ CNN training script
│ ├─ Loads dataset from train/ and test/
│ ├─ Applies image preprocessing & augmentation
│ ├─ Trains the deep learning model
│ └─ Saves the trained model as .h5
│
├── webcam.py
│ ├─ Loads the trained model
│ ├─ Uses OpenCV for face detection
│ ├─ Captures real-time webcam frames
│ └─ Predicts and displays emotions live
│
├── face_emotion_model.h5
│ └─ Saved trained CNN model
│
├── train/
│ └─ Training images organized by emotion
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── neutral/
│ ├── sad/
│ └── surprise/
│
└── test/
└─ Testing images (same emotion classes)
```

---

## 🔧 What I Did in This Project

- Designed and trained a **CNN architecture** for facial emotion recognition  
- Used **ImageDataGenerator** for preprocessing and data augmentation  
- Organized emotion-wise datasets for supervised learning  
- Implemented **real-time emotion detection** using OpenCV and webcam input  
- Integrated **Haar Cascade face detection** to isolate faces before prediction  
- Saved and reused the trained model for inference  

This project combines **computer vision**, **deep learning**, and **real-time processing** into a single pipeline.

---

## ⚙️ Technologies Used

- **Python**
- **TensorFlow / Keras** – Model training and inference  
- **OpenCV** – Face detection & webcam handling  
- **NumPy** – Numerical processing  

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install tensorflow opencv-python numpy
```

### 2️⃣ Train the Emotion Detection Model
```bash
python Face_reactions.py
```

### 3️⃣ Run Real-Time Emotion Detection
```bash
python webcam.py
```



