import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd

from actions import actions

DATA_PATH = os.path.join('datas_raw')
#규칙 : 왼손사용 / 반복동작은 한번만 / 촬영각도와 손 풀림정도, 간격 등만 바꿔가며하기

sequence_count = 20
sequence_length = 30

for action in actions:
    try:
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass
        
        

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

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    
    while capture.isOpened():
        for action in actions:
            for sequence in range(20, 20+sequence_count):
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
                for frame_num in range(sequence_length):
                    ret, frame = capture.read()
                    image, results = mp_hand_detection(frame, hands)
                    
                    cv2.putText(image, f'Collecting frames for {action} - {sequence+1}', (15,12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 1, cv2.LINE_AA)

                    draw_landmarks(image, results)
                    cv2.imshow('Data Recorder', image) 
                    hand = extract_keypoints(results)
                    sequence_datas = np.append(sequence_datas, np.reshape(hand, (1, -1)), axis = 0)
                    cv2.waitKey(10)
                sequence_path = os.path.join(DATA_PATH, action, str(sequence)+".tsv")
                pd.DataFrame(sequence_datas).to_csv(sequence_path,sep='\t', index = False) 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()
