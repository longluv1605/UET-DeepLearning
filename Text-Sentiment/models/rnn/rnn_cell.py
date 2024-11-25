import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        if self.nonlinearity != "tanh" and self.nonlinearity != "relu":
            raise ValueError("Invalid nonlinearity!")

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
            hx = torch.zeros(input.size(0), self.hidden_size)

        y = self.x2h(x) + self.h2h(hx)

        if self.nonlinearity == "tanh":
            y = torch.tanh(y)
        else:
            y = torch.tanh(y)

        return y


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(LSTMCell, self).__init__()
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, ht_1=None):
        if ht_1 is None:
            ht_1 = torch.zeros(x_t.size(0), self.hidden_size)
            ht_1 = (ht_1, ht_1)

        ht_1, ct_1 = ht_1

        gates = self.x2h(x_t) + self.h2h(ht_1)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        f_t = torch.sigmoid(forget_gate)
        i_t = torch.sigmoid(input_gate)
        c_t_d = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = ct_1 * f_t + i_t * c_t_d

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
