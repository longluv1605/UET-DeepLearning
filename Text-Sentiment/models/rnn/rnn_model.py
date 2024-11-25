import torch
import torch.nn as nn
from rnn_cell import RNNCell, LSTMCell

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_size, emb_matrix=None, dropout=0.5, bias=True, activation='tanh'):
        super(SimpleRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias =  bias
        
        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        
        
        if activation == 'tanh':
            self.rnn_cell_list = RNNCell(input_size=emb_dim,
                               hidden_size=hidden_size,
                               bias=bias,
                               nonlinearity='tanh')
        elif activation == 'relu':
            self.rnn_cell_list = RNNCell(input_size=emb_dim,
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
        #       x: of shape (batch_size, seq_length)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       y: of shape (batch_size, output_size)
        
        
        x = self.embedding(x) # of shape (batch_size, seq_length, emb_dim)
        
        if hx is None:
            if torch.cuda.is_available():
                hx = torch.zeros(x.size(0), self.hidden_size).cuda()
            else:
                hx = torch.zeros(x.size(0), self.hidden_size)
        
        # Forward by time
        for t in range(x.size(1)):
            hx = self.rnn_cell_list(x[:, t, :], hx)
        
        y = self.clf(hx)
        
        return y
    
    
class StageRNN(SimpleRNN):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_size, num_layers=1, emb_matrix=None, dropout=0.5, bias=True, activation='tanh'):
        super().__init__(vocab_size, emb_dim, hidden_size, output_size, emb_matrix, dropout, bias, activation)
        
        self.num_layers = num_layers
        
        self.rnn_cell_list = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.rnn_cell_list.append(RNNCell(emb_dim, hidden_size, bias, activation))
            else:
                self.rnn_cell_list.append(RNNCell(hidden_size, hidden_size, bias, activation))
                
    
    def forward(self, x, hx=None):
        x = self.embedding(x)
        
        if hx is None:
            if torch.cuda.is_available():
                hx = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
            else:
                hx = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
        outs = []
        hidden = []
        for layer in range(self.num_layers):
            hidden.append(hx[layer, :, :])

        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](x[:, t, :], hx[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hx[layer - 1], hx[layer])
                    
                hidden[layer] = hidden_l
            
            outs.append(hidden_l)
        
        y = outs[-1].squeeze()
        y = self.clf(y)
        
        return y
    

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_size, num_layers=1, emb_matrix=None, dropout=0.5, bias=True, activation='tanh'):
        super(LSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias =  bias
        self.num_layers = num_layers
        
        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
            
        self.lstm_cell_list = nn.ModuleList()
        
        self.lstm_cell_list.append(LSTMCell(emb_dim, hidden_size))
        for _ in range(1, num_layers):
            self.lstm_cell_list.append(LSTMCell(hidden_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x, h0=None):
        x = self.embedding(x) # of shape (batch_size, seq_length, emb_dim)
        
        if h0 is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
            else:
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        
        outs = []
        hidden = []
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))
        
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.lstm_cell_list[layer](x[:, t, :], hidden[layer][0])
                else:
                    hidden_l = self.lstm_cell_list[layer](hidden[layer - 1][0], hidden[layer])
                
                hidden[layer] = hidden_l
                
            outs.append(hidden_l[0])
        
        out = outs[-1].squeeze()
        out = self.clf(out)
        
        return out
                