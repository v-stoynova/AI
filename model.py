import os
import cv2
import numpy as np
from model import Model

# Set environment variable to limit TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Load the trained model
cnn_model = Model()

# Prevent usage of OpenCL to avoid potential issues and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary to map numerical labels to emotions
emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        # Format the ROI for the model
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (224, 224))
        # cropped_img = np.expand_dims(cropped_img, axis=-1)  # Get shape: (244, 244, 1)
        cropped_img = np.expand_dims(cropped_img, axis=0)  # Shape should be (1, 224, 224, 1)

        # Convert to float and scale the image
        cropped_img = cropped_img.astype('float32') / 255.0

        # Predict emotion and assign label
        prediction_idx = cnn_model.predict(cropped_img)
        emotion_label = emotions[prediction_idx]
        cv2.putText(frame, emotion_label, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Emotion Detector", frame)

    # Break the loop when 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
