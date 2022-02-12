import os
import cv2
import numpy as np

from model import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Load the train model
cnn_model = Model()

# Prevent openCL usage and unecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dict which assigns each label an emotion (ASC)
emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h  + 10), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (244, 244)), -1), 0)

        prediction_indx = cnn_model.predict(cropped_img)

        # Display the emotion on your face
        cv2.putText(frame, emotions[prediction_indx], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        

    cv2.imshow("Emotion Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
       break

cap.release()

cv2.destroyAllWindows()
