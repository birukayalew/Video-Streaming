import numpy as np
import dlib
import cv2
from keras.models import load_model
from imutils import face_utils
from django.shortcuts import render
import os
import csv
import time
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DIR + '/server/models/shape_predictor_68_face_landmarks.dat')
emotion_classifier = load_model(DIR + '/server/models/_mini_XCEPTION.102-0.66.hdf5', compile=False)

(lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(i_lower, i_upper) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
(el_lower, el_upper) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(er_lower, er_upper) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def index(request):
    return render(request, 'server/index.html')

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
def get_emotion(faces, frame):
    try:
        x, y, w, h = face_utils.rect_to_bb(faces)
        roi = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
        roi = roi / 255.0
        roi = np.array([roi]) 

        preds = emotion_classifier.predict(roi)[0]
        # emotion_probability = np.max(preds)
        emotion = EMOTIONS[preds.argmax()]

        return emotion
    except:
        pass
    
def clear_csv():
    with open(DIR+'/server/static/server/data/fps.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FPS"])
        f.close()