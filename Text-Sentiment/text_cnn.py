import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional Layers with different kernel size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layer
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # Get embedding
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length, embedding_dim)

        # Apply Conv
        conv_results = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # (batch_size, num_filters, seq_len - k + 1)
        pooled_results = [torch.max(result, dim=2)[0] for result in conv_results]  # Max pooling

        # Concate result from kernel sizes
        x = torch.cat(pooled_results, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        
        x = self.dropout(x)
        
        # Fully connected layer to classification
        x = self.fc(x)  # (batch_size, num_classes)
        return x
