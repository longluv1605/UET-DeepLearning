import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        if self.nonlinearity != 'tanh' and self.nonlinearity != 'relu':
            raise ValueError('Invalid nonlinearity!')
        
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, hx=None):
        # Input
        #       x: of shape (batch_size, input_size)
        #
        #       hx: of shape (batch_size, hidden_size)
        #
        # Output
        #       y: of shape (batch_size, hidden_size)
        
        if hx is None:
            if torch.cuda.is_available():
                hx = torch.zeros(x.size(0), self.emb_dim).cuda()
            else:
                hx = torch.zeros(x.size(0), self.emb_dim)
            
        y = self.x2h(x) + self.h2h(hx)
        
        if self.nonlinearity == 'tanh':
            y = nn.Tanh(y)
        else:
            y = nn.ReLU(y)
            
        return y
    

class LSTM_Cell(nn.Module):
    pass