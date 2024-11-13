import torch
from utils import *

def initialize_lossfn_and_optimizer(model: torch.nn.Module):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = 0.001)
    return loss_fn, optimizer

def train_model(
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 3,
    device=None,
    seed=42,
    write_log=False
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if write_log != False:
        setup_logger(log_file=f'logs/{write_log}.log')
        logging.info("Training started\n")
    
    model.to(device)
    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = 0
        for i, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader)

        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                loss = loss_fn(y_logits, y)
                test_loss += loss
                y_pred = torch.argmax(y_logits, dim=1)
                correct = torch.eq(y_pred, y).sum().item()
                acc = correct / len(y_pred) * 100
                test_acc += acc

            test_loss /= len(testloader)
            test_acc /= len(testloader)

        print(
            f"\tTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n"
        )
        if write_log:
            logging.info(f"\nEpoch [{epoch+1}/{epochs}] -->\tTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")


def save_model(model, name):
    filepath = 'models/' + name + '.pth'
    torch.save(model.state_dict(), filepath)