import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('face_emotion_model.h5')

# If you know the classes (labels) in the order they were generated:
# E.g. train_generator.class_indices might look like: {'angry': 0, 'happy': 1, 'sad': 2, ...}
# So we invert that dictionary to get class labels by index
class_labels = ["angry", "happy", "sad", "neutral", "fear", "surprise", "disgust"]  
# Make sure to match exactly the classes in your dataset and in the correct index order.

# Haar Cascade for face detection (ensure you have this .xml file in your working directory)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Desired size (must match your model's input)
IMG_HEIGHT = 48
IMG_WIDTH = 48

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break  # if there's an error in reading the frame, just exit

    # Convert to grayscale (since our model expects grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face (optional)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize and reshape for model input
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.reshape(roi_gray, (1, IMG_HEIGHT, IMG_WIDTH, 1))
        
        # Prediction
        preds = model.predict(roi_gray)
        emotion_label = class_labels[np.argmax(preds)]

        # Put text of emotion on the frame
        cv2.putText(
            frame, 
            emotion_label, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 2
        )

    # Show the frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
