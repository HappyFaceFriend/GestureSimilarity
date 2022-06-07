from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import random
from tqdm import tqdm
from math import ceil
import pandas as pd
from model import get_model
import train_settings

MODEL_PATH = os.path.join('models', 'mark14-2e3-256_128')
PT_MODEL_PATH = os.path.join('models_pt', 'mark12-1e2-b32-layerdiff')

train_test_ratio = 0.9
batch_size = 1024
learning_rate = 2e-3
num_epoch = 80

SAME = 1
DIFF = 0

try:
    os.makedirs(MODEL_PATH)
except:
    pass
DATA_PATH = os.path.join('datas_preprocessed')

open(os.path.join(MODEL_PATH, 'train_settings.txt'),'w').write(str({'batch_size':batch_size, 'lr':learning_rate, 'ep':num_epoch}))

datas = {}
for action in train_settings.actions:
    datas[action] = []
    for sequence in range(train_settings.sequence_count):
        sequence_path = os.path.join(DATA_PATH, action, str(sequence)+".tsv")
        sequence_data = pd.read_csv(sequence_path, sep='\t', header=0)
        datas[action].append(torch.Tensor(sequence_data.values))

def shuffle(target_list):
    for i in range(5):
        random.shuffle(target_list)

train_list, test_list = [], []
same_list, diff_list = [], []
print("Splitting and arranging datas...")
for action in tqdm(datas.keys()):
    same_pairs, diff_pairs = [], []
    for i in range(len(datas[action])):
        for j in range(i+1, len(datas[action])):
            same_pairs.append(((datas[action][i], datas[action][j]), SAME))
    shuffle(same_pairs)
    for other in datas.keys():
        this_pairs = []
        if action == other:
            continue
        for i in range(len(datas[action])):
            for j in range(len(datas[other])):
                this_pairs.append(((datas[action][i], datas[other][j]), DIFF))
        shuffle(this_pairs)
        diff_pairs.extend(this_pairs[:ceil(len(same_pairs)  / (len(datas.keys())-1))])
    
    train_list.extend(same_pairs[:int(train_test_ratio * len(same_pairs))])
    train_list.extend(diff_pairs[:int(train_test_ratio * len(diff_pairs))])

    test_list.extend(same_pairs[int(train_test_ratio * len(same_pairs)):])
    test_list.extend(diff_pairs[int(train_test_ratio * len(diff_pairs)):])

shuffle(train_list)
shuffle(test_list)


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

model = get_model()
optimizer = optim.AdamW(model.embedding.parameters(), lr = learning_rate)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = num_epoch/10, gamma = 0.6)
criterion = nn.BCELoss()

best_accuracy = 0
losses = []
accuracies = []
f1s = []
model.to(device=cuda)
for epoch in range(num_epoch):
    #Training
    model.train()
    running_loss = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        (data1, data2), label = batch
        data1, data2, label = data1.to(device=cuda), data2.to(device=cuda), label.to(device=cuda).type(torch.float32)
        output = model(data1, data2)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    lr_scheduler.step()
    print(f"Epoch {epoch} : Train Loss {running_loss}")
    #Eval
    model.eval()
    trues, preds = [], []
    for batch in tqdm(test_dataloader):
        (data1, data2), label = batch
        data1, data2, label = data1.to(device=cuda), data2.to(device=cuda), label.to(device=cuda)
        sim = model(data1, data2)
        trues += label.to(device=cpu).tolist()
        preds += sim.to(device=cpu).tolist()

    accuracy = len([1 for i in range(len(trues)) if trues[i] == SAME and preds[i] > (SAME+DIFF)/2 or trues[i]==DIFF and preds[i]<(SAME+DIFF)/2]) / len(trues)
    print(f"Epoch {epoch} : Accuracy {accuracy}")
    f1 = f1_score(trues, [SAME if preds[i]>(SAME+DIFF)/2 else DIFF for i in range(len(preds))])
    print(f"Epoch {epoch} : F1 {f1}")
    print()
    accuracies.append(accuracy)
    f1s.append(f1)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.to(device=cpu)
        torch.save(model.embedding.state_dict(), os.path.join(MODEL_PATH, 'model_states_best.pt'))
        model.to(device=cuda)
    open(os.path.join(MODEL_PATH,'train_losses.txt'),'w').write(str(losses))
    open(os.path.join(MODEL_PATH,'accuracies.txt'),'w').write(str(accuracies))
    open(os.path.join(MODEL_PATH,'f1s.txt'),'w').write(str(f1s))
torch.save(model.embedding.state_dict(), os.path.join(MODEL_PATH, 'model_states_last.pt'))
    
    