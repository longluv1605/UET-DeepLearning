import torch.nn as nn

class CNN(nn.Module):
  def __init__(self,
               input_shape : int,
               output_shape: int,
               hidden_units: int):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,
                  out_channels = hidden_units,
                  kernel_size = 2,
                  padding = 1,
                  stride = 1),  # (B, hidden_units, [(input - kernel + 2padding) / stride + 1])
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 2,
                  padding = 1,
                  stride = 1),  # (B, hidden_units, [(input - kernel + 2padding) / stride + 1])
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,
                     stride = 2)  # (B, hidden_units, [(input - kernel) / stride + 1])
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 2,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size = 2,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,
                     stride = 2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 10*9*9,
                  out_features = output_shape)
    )
  def forward(self, x):
    return self.classifier(self.block2(self.block1(x)))