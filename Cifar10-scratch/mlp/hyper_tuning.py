from itertools import product
from torch.optim import Adam, SGD

from model_trainer import ModelTrainer
from dataloader_factory import DataLoaderFactory

def hyper_tuning(model, epochs=3, write_log=False, download=True):
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [1e-3, 5e-3],
        'optimizer': ['SGD', 'Adam']
    }

    param_combinations = list(product(*param_grid.values()))

    best_params = None
    best_loss = float('inf')
    for params in param_combinations:
        batch_size, lr, opt = params
        print(f"\nTesting with batch size={batch_size}, learning rate={lr}, optimizer={opt}")
        
        dataloader = DataLoaderFactory(root='./data', batch_size=batch_size, download=download)

        optimizer = SGD(model.parameters(), lr=lr) if opt == 'SGD' else Adam(model.parameters(), lr=lr)

        model_trainer = ModelTrainer(model=model, dataloader=dataloader, optimizer=optimizer)
        avg_loss = model_trainer.train(epochs=epochs, write_log=write_log)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'optimizer': opt
            }

    return best_params, best_loss, model