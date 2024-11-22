from data_loader import load_data
from model_controller import train
from text_cnn import TextCNN

import torch.nn as nn

from itertools import product
from torch.optim import Adam, SGD



# Fine tuning model
def hyper_tuning(dataframe, epochs=5):
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [1e-3, 5e-3],
        'optimizer': ['SGD', 'Adam'],
        'embedding_dim': [50, 100, 150],
        'num_filters': [50, 100, 150]
    }

    param_combinations = list(product(*param_grid.values()))
    
    best_params = None
    best_loss = float('inf')
    for params in param_combinations:
        batch_size, lr, opt, emb_dim, num_fils = params
        print(f"\nTesting with batch size={batch_size}, learning rate={lr}, optimizer={opt}, embedding_dim={emb_dim}, num_filters={num_fils}")
        
        vocab, train_loader, _ = load_data(dataframe, batch_size=batch_size)
        
        vocab_size = len(vocab)
        embedding_dim = emb_dim
        num_classes = 2  # Positive/Negative
        kernel_sizes = [3, 4, 5]
        num_filters = num_fils

        model = TextCNN(vocab_size, embedding_dim, num_classes, kernel_sizes, num_filters)

        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=lr) if opt == 'SGD' else Adam(model.parameters(), lr=lr)

        avg_loss = train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, epochs=epochs)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'optimizer': opt,
                'embedding_dim': emb_dim,
                'num_filters': num_fils
            }

    return best_params, best_loss