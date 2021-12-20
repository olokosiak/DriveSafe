import os
import cv2 # real time facce detection library
import numpy as np # image manipulation
from keras.preprocessing import image # library for image processing
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("DriveSafe/best_model.h5") # Loading in a pre-trained model, according to image data using deep learning, which containts layers of image datasets used for face recognition. 
# Emotion prediction

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Machine learning object detection(face) that identifies objects in stream


cap = cv2.VideoCapture(0) # displays camera feed

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) # Converts BGR to RGB so model can detect

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5) # If faces found, return position of face as rectangle

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7) # Settomg rectangle to out liking
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224)) # Resizing roi to 224*224 pixels
        img_pixels = image.img_to_array(roi_gray) # Converts image into array for manipulation. Changing the shape of the array from 0-255 into single row and multiple columns[0,1]
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels) # Predicting emotion using model

        max_index = np.argmax(predictions[0]) # find max indexed array, picking out the most common emotion index

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index] # Returning most indexed emotion

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Displays value for emotion in roi

    resized_img = cv2.resize(test_img, (1000, 700)) # Image display size
    cv2.imshow('Facial emotion analysis ', resized_img) # Displays window

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows