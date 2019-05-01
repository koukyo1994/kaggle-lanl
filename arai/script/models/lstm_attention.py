import torch.nn as nn

from arai.script.models.layers import Attention


class LSTMAttentionNet(nn.Module):
    def __init__(self, hidden_size, linear_size, input_shape, n_attention):
        super(LSTMAttentionNet, self).__init__()

        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.lstm1 = nn.LSTM(
            self.input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hidden_size * 2,
            int(hidden_size / 2),
            bidirectional=True,
            batch_first=True)
        self.attn = Attention(
            int(hidden_size / 2) * 2, self.maxlen, n_attention, n_attention)
        self.lin1 = nn.Linear(int(hidden_size / 2) * 2, linear_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(linear_size, 1)

    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        h_lstm2, _ = self.lstm2(h_lstm1)
        attn = self.attn(h_lstm2)
        lin = self.relu(self.lin1(attn))
        out = self.lin2(lin)
        return out
