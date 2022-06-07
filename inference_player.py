import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
from inference_actions import actions
from preprocess import get_preprocessed
from model import get_model
import torch

PRESET_DATA_PATH = os.path.join('inference_datas','preset')
#규칙 : 왼손사용 / 반복동작은 한번만 / 촬영각도와 손 풀림정도, 간격 등만 바꿔가며하기

mode = 'SINGLE'
sequence_count = 3
presets = []
for action in actions:
    for sequence in range(sequence_count):
        presets.append((torch.load(os.path.join(PRESET_DATA_PATH, action, str(sequence)+".pt")), action))

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

model = get_model(os.path.join('models','mark14-2e3-256_128','model_states.pt'))
model.eval()
embeding_model = model.embedding
cosine = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
def get_logits(raw_input):
    preprocessed = get_preprocessed(raw_input)
    input = torch.tensor(preprocessed, dtype = torch.float32)
    embedding = embeding_model(input)
    logits = []
    for preset, label in presets:
        similarity = (1 + cosine(embedding, preset)) / 2
        logits.append((label, round(similarity.item(), 3)))
    logits.sort(key = lambda x: x[1], reverse = True)
    return logits

stride = [0.7,1,1.5, 2,2.5]
threshold = 0.9
class FrameCollection:
    def __init__(self, stride):
        self.frames = np.empty((int(stride * 10), 63))
        self.stride = stride
        self.stack = 0
        self.logits = None
    def tick(self, frame):
        self.frames = np.append(self.frames, np.reshape(frame, (1, -1)), axis = 0)
        self.frames = np.delete(self.frames, 0, axis = 0)
        self.logits = get_logits(self.frames)
    def best_logit(self):
        if self.logits == None:
            return ['None',0]
        else:
            return self.logits[0]
frame_collections = [FrameCollection(stride[i]) for i in range(len(stride))]
            

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    
    while capture.isOpened():
        while True:
            ret, frame = capture.read()
            image, results = mp_hand_detection(frame, hands)
            
            draw_landmarks(image, results)
            hand = extract_keypoints(results)
            for frame_collection in frame_collections:
                frame_collection.tick(hand)
            if mode == 'LOGITS':
                for i in range(len(frame_collections)):
                    best_logit = frame_collections[i].best_logit()
                    thickness = 1
                    if best_logit[1] > threshold:
                        thickness= 2
                    cv2.putText(image, f'stride{frame_collections[i].stride}: {best_logit[0]} {best_logit[1]}', 
                        (15,12+i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), thickness, cv2.LINE_AA)
            else:
                best_logit = max([col.best_logit() for col in frame_collections], key=lambda x:x[1])
                if best_logit[1] > threshold:
                    cv2.putText(image, f'{best_logit[0]}', 
                        (15,30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow('Data Recorder', image) 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    capture.release()
    cv2.destroyAllWindows()
