import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, n_hidden_nodes, n_classes, image_width, image_height, color_channels):
        super().__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_classes = n_classes
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU6()
        
        self.image_width = image_width
        self.image_height = image_height
        self.color_channels = color_channels
        self.input_size = self.image_width * self.image_height * self.color_channels
        
        # define layers
        self.hidden_layer = nn.Linear(self.input_size, self.n_hidden_nodes)
        self.output_layer = nn.Linear(self.n_hidden_nodes, self.n_classes)
        
    def forward(self, x):
        x = x.view(-1, self.image_width, self.input_size)
    
        x = self.relu(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer())
        
        return x
