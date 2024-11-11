import torch.nn as nn

# Define MLP model
class MLP(nn.Module):
    def __init__(self, n_hidden_nodes, n_classes, image_width=32, image_height=32, color_channels=3, n_hidden_layers=1):
        super(MLP, self).__init__()
        input_size = image_width * image_height * color_channels
        self.layers = nn.Sequential(
            nn.Linear(input_size, n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(n_hidden_nodes, n_classes)
        )
        
        if n_hidden_layers > 1:
            self.added_layers = nn.Sequential()
            for i in range(n_hidden_layers - 1):
                self.added_layers.add_module(str(2 * (i + 1) + 1), nn.Linear(n_hidden_nodes, n_hidden_nodes))
                self.added_layers.add_module(str(2 * (i + 1) + 2), nn.ReLU())
            layers = list(self.layers)
            layers.insert(2, self.added_layers)
            self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
