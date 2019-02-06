import torch
import torch.nn as nn


class NeuralNet_2(nn.Module):
    def __init__(self):
        super(NeuralNet_2, self).__init__()

        hidden_size = 64

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 8 + train_features.shape[1], hidden_size * 8 + train_features.shape[1]),
            nn.BatchNorm1d(hidden_size * 8 + train_features.shape[1]),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 8 + train_features.shape[1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.linear = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x[0])
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        # [512 x 368]
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, f), 1)
        conc = self.fc1(conc)
        conc = self.fc2(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
