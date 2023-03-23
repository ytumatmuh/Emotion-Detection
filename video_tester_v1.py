import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
#from keras.preprocessing.image import load_img, img_to_array 
#from tensorflow.keras.utils import load_img
#from tensorflow.keras.utils import img_to_array
from keras.utils import load_img, img_to_array
import keras.utils as image
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
test_img = cv2.imread("fear.jpeg")
gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

for (x, y, w, h) in faces_detected:
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
    roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    roi_gray = cv2.resize(roi_gray, (224, 224))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow('Facial emotion analysis ', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows