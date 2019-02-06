import torch
import torch.nn as nn


class NeuralNet_3(nn.Module):
    def __init__(self):
        super(NeuralNet_3, self).__init__()

        hidden_size = 64

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.conv = nn.Conv1d(maxlen, 48, 4, padding=1)
        self.pool_a = nn.AvgPool1d(2)
        self.pool_m = nn.MaxPool1d(2)

        self.fc1 = nn.Sequential(nn.Linear(12096, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())

        self.linear = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        bs = x.shape[0]

        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_conv = self.conv(h_lstm)
        h_l_a = self.pool_a(h_lstm_conv)
        h_l_m = self.pool_m(h_lstm_conv)

        h_gru_conv = self.conv(h_gru)
        h_g_a = self.pool_a(h_gru_conv)
        h_g_m = self.pool_m(h_gru_conv)

        conc = torch.cat((h_l_a.view(bs, -1), h_l_m.view(bs, -1), h_g_a.view(bs, -1), h_g_m.view(bs, -1)), 1)
        # print(conc.shape)
        conc = self.fc1(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
