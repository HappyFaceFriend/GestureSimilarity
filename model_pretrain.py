from sched import scheduler
from tkinter import Variable
from sklearn.metrics import f1_score
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import random
from tqdm import tqdm
from math import ceil

from actions import actions
import pandas as pd
import numpy as np

from model import get_class_model

MODEL_PATH = os.path.join('models_pt', 'mark12-1e2-b32-layerdiff')

batch_size = 32
learning_rate = 1e-2
num_epoch = 2000

try:
    os.makedirs(MODEL_PATH)
except:
    pass
DATA_PATH = os.path.join('datas_preprocessed')

open(os.path.join(MODEL_PATH, 'train_settings.txt'),'w').write(str({'batch_size':batch_size, 'lr':learning_rate, 'ep':num_epoch}))

sequence_count = 50

datas = {}
for action in actions:
    datas[action] = []
    for sequence in range(sequence_count):
        sequence_path = os.path.join(DATA_PATH, action, str(sequence)+".tsv")
        sequence_data = pd.read_csv(sequence_path, sep='\t', header=0)
        datas[action].append(torch.Tensor(sequence_data.values))

def shuffle(target_list):
    for i in range(5):
        random.shuffle(target_list)

train_list, test_list = [], []
train_test_ratio = 0.9

same_list, diff_list = [], []
print("Splitting and arranging datas...")
label = 0
for action in tqdm(datas.keys()):   
    total_list = []
    for i in range(len(datas[action])):
        total_list.append((datas[action][i], label))
    shuffle(total_list)
    train_list.extend(total_list[:int(train_test_ratio * len(total_list))])
    test_list.extend(total_list[int(train_test_ratio * len(total_list)):])
    label += 1

cuda = torch.device('cuda')
cpu = torch.device('cpu')


class HandDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, index):
        return self.pairs[index]


train_dataset = HandDataset(train_list)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = HandDataset(test_list)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


model = get_class_model(20)
optimizer = optim.SGD([
    {'params':model.embedding.parameters()},
    {'params':model.fc_c.parameters(), 'lr' : learning_rate/10}
], lr=learning_rate, momentum=0.9)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = num_epoch/10, gamma = 0.6)

crit = nn.CrossEntropyLoss()
best_accuracy = 0
losses = []
accuracies = []
model.to(device=cuda)
for epoch in range(num_epoch):
    #Training
    model.train()
    running_loss = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        data, label = batch
        data, label = data.to(device=cuda), label.to(device=cuda)
        output = model(data)
        loss = crit(output, label).to(device=cuda)
        losses.append(loss.to(device=cpu).item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    lr_scheduler.step()
    print(f"Epoch {epoch} : Train Loss {running_loss}")
    #Eval
    model.eval()
    trues, preds = [], []
    for batch in tqdm(test_dataloader):
        data, label = batch
        data, label = data.to(device=cuda), label.to(device=cuda)
        output = model(data)
        trues += label.to(device=cpu).tolist()
        preds += torch.argmax(output, dim = 1).to(device=cpu).tolist()
    print(trues[:10])
    print(preds[:10])
    
    accuracy = len([1 for i in range(len(trues)) if trues[i] == preds[i]]) / len(trues)
    print(f"Epoch {epoch} : Accuracy {accuracy}")
    f1 = f1_score(trues, preds, average='macro')
    print(f"Epoch {epoch} : F1 {f1}")
    print()
    accuracies.append(f1)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.to(device=cpu)
        torch.save(model.embedding.state_dict(), os.path.join(MODEL_PATH, 'model_states.pt'))
        model.to(device=cuda)
    open(os.path.join(MODEL_PATH,'train_losses.txt'),'w').write(str(losses))
    open(os.path.join(MODEL_PATH,'accuracies.txt'),'w').write(str(accuracies))
    
    