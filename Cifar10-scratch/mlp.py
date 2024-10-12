import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, n_hidden_nodes, n_classes, image_width=32, image_height=32, color_channels=3):
        super().__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_classes = n_classes
        
        self.relu = nn.ReLU6()
        self.leaky_relu = nn.LeakyReLU()
        
        self.image_width = image_width
        self.image_height = image_height
        self.color_channels = color_channels
        self.input_size = self.image_width * self.image_height * self.color_channels
        
        # define layers
        self.hidden_layer = nn.Linear(self.input_size, self.n_hidden_nodes) # (B, c * w * h) -> (B, 100)
        self.hidden_layer_2 = nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes) # (B, 100) -> (B, 100)
        self.output_layer = nn.Linear(self.n_hidden_nodes, self.n_classes) # (B, 100) -> (B, nclasses)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden_layer(x))
        x = self.leaky_relu(self.hidden_layer_2(x))
        x = self.leaky_relu(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x

