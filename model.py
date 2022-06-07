
import torch
import torch.nn as nn
import torch.nn.functional as F

margin = 0
input_size = 21 * 3
hidden_size = 256

class EmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(EmbeddingModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, 
                                dropout = dropout, batch_first = True)
        self.fc_a = nn.Linear(hidden_size, 128)

    def forward(self, x):
        #input shape: batch, seq, dim
        _, (hidden_out, _) = self.lstm(x)
        x = hidden_out.view(-1, self.hidden_size) # Reshaping the data for starting LSTM network
        x = self.fc_a(x)
        return x

class Classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(Classification, self).__init__()
        self.embedding = EmbeddingModel(input_size, hidden_size, 0)
        self.fc_c = nn.Linear(128, num_labels)

    def forward(self, x):
        #input shape: batch, seq, dim
        x = self.embedding(x)
        x = self.fc_c(x)
        x = F.softmax(x, dim = 1)
        return x

class BiEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiEncoderModel, self).__init__()
        self.embedding = EmbeddingModel(input_size, hidden_size, 0)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cosineloss = nn.CosineEmbeddingLoss(margin = margin, reduction = 'mean')
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()

    def forward(self, dataA, dataB, label = None):
        #input shape: batch, seq, dim
        outputA = self.embedding(dataA)
        outputB = self.embedding(dataB)
        #print(outputA.size())
        out = (1 + self.similarity(outputA, outputB)) / 2
        
        if label is None:
            return out
        else:
            return out

def get_model():
    model = BiEncoderModel(input_size, hidden_size)
    return model

def get_model(model_path):
    model = BiEncoderModel(input_size, hidden_size)
    model.embedding.load_state_dict(torch.load(model_path))
    return model

def get_class_model(num_labels):
    model = Classification(input_size, hidden_size, num_labels)
    return model