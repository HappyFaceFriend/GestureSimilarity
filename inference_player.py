import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
from preprocess import get_preprocessed
from model import get_embedding_model, get_cosine_similarity
import torch
import inference_settings

from mediapipe_utils import draw_landmarks, extract_keypoints, mp_hand_detection

PRESET_DATA_PATH = os.path.join('inference_datas','preset')

actions = inference_settings.actions
sequence_counts = inference_settings.sequence_counts
sample_length = inference_settings.sample_length

mode = 'LOGITS' #'SINGLE' to print one action / 'LOGITS' to print logits

stride = [0.7,1,1.5, 2,2.5]
threshold = 0.9

presets = []
for action_index in range(len(actions)):
    action = actions[action_index]
    for sequence in range(sequence_counts[action_index]):
        presets.append((torch.load(os.path.join(PRESET_DATA_PATH, action, str(sequence)+".pt")), action))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
capture = cv2.VideoCapture(0)

embedding_model = get_embedding_model(os.path.join('models','mark14-2e3-256_128','model_states.pt'))
embedding_model.eval()

def get_logits(raw_input):
    preprocessed = get_preprocessed(raw_input)
    input = torch.tensor(preprocessed, dtype = torch.float32)
    embedding = embedding_model(input)
    logits = []
    for preset, label in presets:
        similarity = (1 + get_cosine_similarity(embedding, preset)) / 2
        logits.append((label, round(similarity.item(), 3)))
    logits.sort(key = lambda x: x[1], reverse = True)
    return logits

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
