import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
from tqdm import tqdm
import math
import train_settings

def get_frame(sequence_data, frame):
    min, max = math.floor(frame), math.ceil(frame)
    return (sequence_data[max] - sequence_data[min]) * (frame - min) + sequence_data[min]

def get_preprocessed(sequence_data, sample_length = train_settings.sample_length):
    #sample
    value = np.empty((0,len(sequence_data[0])))
    step = (len(sequence_data)-1) / (sample_length-1)
    value = np.append(value, np.reshape(sequence_data[0], (1,-1)), axis = 0)
    for frame in range(1, sample_length-1):
        rate = frame * step
        value = np.append(value, np.reshape(get_frame(sequence_data, rate), (1,-1)), axis = 0)
    value = np.append(value, np.reshape(sequence_data[-1], (1,-1)), axis = 0)
    
    #normalize
    initial = value[0][0]
    maxX, maxY, maxZ, minX, minY, minZ = [initial for x in range(6)]

    for f in range(len(value)):
        for k in range(0, len(value[f]), 3):
            minX = min(value[f][k], minX)
            maxX = max(value[f][k], maxX)
            minY = min(value[f][k+1], minY)
            maxY = max(value[f][k+1], maxY)
            minZ = min(value[f][k+2], minZ)
            maxZ = max(value[f][k+2], maxZ)

    for f in range(len(value)):
        for k in range(0, len(value[f]), 3):
            value[f][k] = (value[f][k] - minX) / (maxX - minX)
            value[f][k+1] = (value[f][k+1] - minY) / (maxY - minY)
            value[f][k+2] = (value[f][k+2] - minZ) / (maxZ - minZ)

    
    for f in range(1, len(value)):
        for k in range(1, len(value[f])):
            value[f][k] -= value[f][0]
        value[f][0] -= value[0][0]
    value[0][0] -= value[0][0]
    return value

if __name__ == "__main__":
    DATA_PATH = os.path.join('datas_preprocessed')
    RAW_DATA_PATH = os.path.join('datas_raw')

    sequence_count = train_settings.sequence_count

    for action in tqdm(train_settings.actions):
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except:
            pass
        for sequence in range(sequence_count):
            sequence_path = os.path.join(RAW_DATA_PATH, action, str(sequence)+".tsv")
            sequence_data = pd.read_csv(sequence_path, sep='\t', header=0).to_numpy()
            sequence_data = get_preprocessed(sequence_data)
            new_path = os.path.join(DATA_PATH, action, str(sequence)+".tsv")
            pd.DataFrame(sequence_data).to_csv(new_path, sep='\t', index = False)