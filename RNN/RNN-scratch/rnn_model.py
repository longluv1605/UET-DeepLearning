import torch
import torch.nn as nn
from rnn_cell import RNNCell

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5, bias=True, activation='tanh'):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias =  bias
        
        if activation == 'tanh':
            self.rnn_cell = RNNCell(input_size=input_size,
                               hidden_size=hidden_size,
                               bias=bias,
                               nonlinearity='tanh')
        elif activation == 'relu':
            self.rnn_cell = RNNCell(input_size=input_size,
                                    hidden_size=hidden_size,
                                    bias=bias,
                                    nonlinearity='relu')
        else:
            raise ValueError("Invalid activated function!")
        
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hx=None):
        # Input:
        #       x: of shape (batch_size, seq_length, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       y: of shape (batch_size, output_size)
        
        if hx is None:
            hx = torch.zeros(x.size(0), self.input_size)
        
        # Forward by time
        for t in range(x.size(1)):
            hx = self.rnn_cell(x[:, t, :], hx)
        
        y = self.fc(hx)
        y = self.dropout(y)
        y = self.clf(y)
        
        return y