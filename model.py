import os
import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 21 * 3
hidden_size = 256
embedding_size = 128

class GestureEmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(GestureEmbeddingModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first = True)
        self.fc_a = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        _, (hidden_out, _) = self.lstm(x)
        x = hidden_out.view(-1, self.hidden_size)
        x = self.fc_a(x)
        return x

class BiEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(BiEncoderModel, self).__init__()
        self.embedding_model = GestureEmbeddingModel(input_size, hidden_size, embedding_size)

    def forward(self, dataA, dataB):
        embeddingA = self.embedding(dataA)
        embeddingB = self.embedding(dataB)
        return get_cosine_similarity(embeddingA, embeddingB)

cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
def get_cosine_similarity(embeddingA, embeddingB):
    return (1 + cos_sim(embeddingA, embeddingB)) / 2

def get_model():
    model = BiEncoderModel(input_size, hidden_size)
    return model

def get_embedding_model(model_states_path, input_size = input_size, hidden_size = hidden_size, embedding_size = embedding_size):
    model = GestureEmbeddingModel(input_size, hidden_size, embedding_size)
    model.load_state_dict(torch.load(model_states_path))
    return model