import torch
import torch.nn as nn


class CNN_review(nn.Module):
    def __init__(self, num_embedding, seq_len):
        super(CNN_review, self).__init__()
        self.num_embedding = num_embedding
        embedding_dim = 100

        self.embed = nn.Embedding(num_embedding, embedding_dim=embedding_dim, padding_idx=1)
        self.conv1 = nn.Conv1d(seq_len, embedding_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.p_relu = nn.ReLU()
        self.fc1 = nn.LazyLinear(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv1(x)
        x = self.p_relu(x)
        x = self.conv2(x)
        x = self.p_relu(x)
        x = self.conv3(x)
        x = self.p_relu(x)
        x = self.conv4(x)
        x = self.p_relu(x)
        x = x.view(-1, self.num_embedding)
        x = self.fc1(x)
        x = self.p_relu(x)
        x = self.fc2(x)
        x = self.p_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.p_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

class Embed_CNN_review(nn.Module):
    def __init__(self, seq_len):
        super(Embed_CNN_review, self).__init__()
        embedding_dim = 100

        self.conv1 = nn.Conv1d(seq_len, embedding_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 100)
        self.fc4 = nn.Linear(100, 1)
        self.fc5 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = torch.squeeze(x)
        x = self.sigmoid(self.fc5(x))
        return x

class RNN_review(nn.Module):
    def __init__(self, num_embedding, batch_size=16):
        super(RNN_review, self).__init__()
        self.batch_size = batch_size
        embed_dim = 200
        hid_dim = 200
        num_layers = 16
        dropout_prob = 0.4
        self.embedding = nn.Embedding(num_embedding, embed_dim)  # prepare the lookup table for word embeddings
        self.rnn = nn.LSTM(embed_dim, hid_dim, bias=True, num_layers=num_layers, bidirectional=True,
                           dropout=dropout_prob)  # LSTM 2 layered and bidirectional
        self.fc1 = nn.Linear(hid_dim * 2, 100)  # fully connected layer for output
        self.fc2 = nn.LazyLinear(1)  # fully connected layer for output
        self.dropout = nn.Dropout(p=dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed_out = self.dropout(self.embedding(x))
        rnn_out, (rnn_hid, rnn_cell) = self.rnn(embed_out)

        hidden = self.dropout(rnn_out)
        hidden = self.fc1(hidden)
        hidden = hidden.view(self.batch_size, -1)
        out = self.fc2(hidden)
        return self.sigmoid(out)

class Embed_RNN_review(nn.Module):
    def __init__(self, batch_size=16):
        super(Embed_RNN_review, self).__init__()
        self.batch_size = batch_size
        embed_dim = 100
        hid_dim = 100
        num_layers = 16
        dropout_prob = 0.4
        self.rnn = nn.LSTM(embed_dim, hid_dim, bias=True, num_layers=num_layers, bidirectional=True,
                           dropout=dropout_prob)  # LSTM 2 layered and bidirectional
        self.fc1 = nn.Linear(hid_dim * 2, 100)  # fully connected layer for output
        self.fc2 = nn.LazyLinear(1)  # fully connected layer for output
        self.dropout = nn.Dropout(p=dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rnn_out, (rnn_hid, rnn_cell) = self.rnn(x)
        x = self.dropout(rnn_out)
        x = self.fc1(x)
        x = x.view(self.batch_size, -1)
        x = self.fc2(x)
        return self.sigmoid(x)