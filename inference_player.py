from msilib import sequence
import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
import torch

from model import get_model
from preprocess import get_preprocessed

EMBEDDING_PATH = os.path.join('inference_datas')
MODEL_PATH = os.path.join('models','model_states.pt')
model_states = torch.load(MODEL_PATH)

model = get_model()
model.eval()
model.load_state_dict(model_states)
embedding_model = model.embedding

#규칙 : 왼손사용 / 반복동작은 한번만 / 촬영각도와 손 풀림정도, 간격 등만 바꿔가며하기
actions = [ 'rock', 'scissors', 'paper']

sequence_count = 3
sequence_length = 30

embedding_presets = []
for action in actions:
    for i in range(sequence_count):
        embedding_presets.append((action,torch.load(os.path.join(EMBEDDING_PATH, action, str(i)+".pt"))))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)

def mp_hand_detection(image, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)


def extract_keypoints(results):
    return np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21*3)


sequence_datas = np.empty((30,63))
similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def insert_frame(sequence_datas, hand):
    sequence_datas = np.append(sequence_datas, np.reshape(hand, (1, -1)), axis = 0)
    sequence_datas = np.delete(sequence_datas, 0, axis = 0)
    return sequence_datas



with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        image, results = mp_hand_detection(frame, hands)
        
        draw_landmarks(image, results)
        cv2.imshow('Data Recorder', image) 
        hand = extract_keypoints(results)
        sequence_datas = insert_frame(sequence_datas, hand)
        data = get_preprocessed(sequence_datas)
        embedding = embedding_model(torch.tensor(data, dtype = torch.float32))
        similarities = []
        for action, data in embedding_presets:
            similarities.append((action,similarity(embedding, data).item()))
        similarities.sort(key=lambda x:x[1], reverse = True)
        y = 12
        for action, value in similarities:
            cv2.putText(image, action +":"+str(value), (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            y += 15
        cv2.imshow('Data Recorder', image) 

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()
