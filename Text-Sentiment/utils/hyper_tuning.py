from itertools import product
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

def hyper_tuning(
    model_class,  # Model class (e.g., TextCNN, RNN)
    param_grid,   # Dictionary of hyperparameters
    data_loader_fn,  # DataLoader function (e.g., DataLoader)
    train_fn,  # Training function (e.g., train)
    train_dataset,  # Training dataset
    device=torch.device('cpu'),  # Device to run on
    epochs=5  # Number of epochs
):
    param_combinations = list(product(*param_grid.values()))
    
    best_params = None
    best_loss = float('inf')
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"\nTesting with parameters: {param_dict}")
        
        # Create DataLoader
        train_loader = data_loader_fn(train_dataset, batch_size=param_dict['batch_size'], shuffle=True)
        
        # Initialize model
        model = model_class(**param_dict['model_params'])
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_class = SGD if param_dict['optimizer'] == 'SGD' else Adam
        optimizer = optimizer_class(model.parameters(), lr=param_dict['learning_rate'])

        # Train and calculate loss
        avg_loss = train_fn(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epochs=epochs
        )
        
        # Update best parameters
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = param_dict

    return best_params, best_loss
