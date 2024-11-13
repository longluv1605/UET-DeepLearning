from itertools import product
from torch.optim import Adam, SGD

from dataloader import *
from model_controller import *

def hyper_tuning(model, model_transform, epochs=3, write_log=False):
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
        
        model_train_loader, model_test_loader = get_dataloader(model_transform, batch_size=batch_size)

        optimizer = SGD(model.parameters(), lr=lr) if opt == 'SGD' else Adam(model.parameters(), lr=lr)
        lossfn = torch.nn.CrossEntropyLoss()
        
        avg_loss = train_model(trainloader = model_train_loader,
                            testloader = model_test_loader,
                            model= model,
                            loss_fn = lossfn,
                            optimizer= optimizer,
                            epochs= 3,
                            write_log=write_log)
                            
        if avg_loss < best_loss:    
            best_loss = avg_loss
            best_params = {
                'batch_size': batch_size,
                'learning_rate': lr,
                'optimizer': opt
            }

    return best_params