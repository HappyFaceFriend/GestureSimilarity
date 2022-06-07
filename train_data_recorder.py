import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd

import train_settings
from mediapipe_utils import draw_landmarks, extract_keypoints, mp_hand_detection

DATA_PATH = os.path.join('datas_raw')

actions = train_settings.actions
sequence_count = train_settings.sequence_count
sample_length = train_settings.sample_length

for action in actions:
    try:    os.makedirs(os.path.join(DATA_PATH, action))
    except: pass
        
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
capture = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    while capture.isOpened():
        for action in actions:
            for sequence in range(sequence_count):
                while True:
                    ret, frame = capture.read()
                    image, results = mp_hand_detection(frame, hands)
                    draw_landmarks(image, results)
                    cv2.putText(image, f'Press 5 to Record {action} - {sequence+1}', (20,200), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Data Recorder', image)
                    if (cv2.waitKey(10) & 0xFF == ord('5')):
                        break

                sequence_datas = np.empty((0,63))
                frame_count = 0
                while True:
                    ret, frame = capture.read()
                    image, results = mp_hand_detection(frame, hands)
                    
                    cv2.putText(image, f'Collecting frames for {action} - {sequence+1}:{frame_count}', (15,12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 1, cv2.LINE_AA)

                    draw_landmarks(image, results)
                    cv2.imshow('Data Recorder', image) 
                    hand = extract_keypoints(results)
                    sequence_datas = np.append(sequence_datas, np.reshape(hand, (1, -1)), axis = 0)
                    
                    frame_count += 1
                    if (cv2.waitKey(10) & 0xFF == ord('5')):
                        break
                sequence_path = os.path.join(DATA_PATH, action, str(sequence)+".tsv")
                pd.DataFrame(sequence_datas).to_csv(sequence_path,sep='\t', index = False) 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()
