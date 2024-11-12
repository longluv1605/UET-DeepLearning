import torch
import torchmetrics
import logging
from torch import nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import random

from mlp import MLP
from dataloader_factory import DataLoaderFactory
from log_writer import setup_logger

class ModelTrainer:
    def __init__(self, model=None, criterion=None, optimizer=None, dataloader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or MLP(n_classes=10, n_hidden_nodes=100, image_width=32, image_height=32, color_channels=3)
        self.model.to(self.device)
        self.criterion = criterion or CrossEntropyLoss()
        self.optimizer = optimizer or SGD(self.model.parameters(), lr=0.005)
        self.dataloader = dataloader or DataLoaderFactory(root='./data', batch_size=32, num_workers=4, download=False)
        self.epochs = 0

    def train(self, epochs=10, write_log=False):
        self.epochs = epochs
        train_loader = self.dataloader.load_data(train=True)
        train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
        test_loader = self.dataloader.load_data(train=False)
        test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)

        if write_log:
            num_layers = len(list(self.model.children())) // 2 + 1
            id = random.randint(0, 1000)
            setup_logger(filename=f'training_mlp_{num_layers}_hidden_layers_{id}.log')
            logging.info("Training started\n")

        for epoch in range(epochs):
            running_loss = 0.0
            self.model.train()
            train_accuracy.reset()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                train_accuracy.update(outputs.argmax(dim=1), labels)

            final_train_accuracy = train_accuracy.compute()
            print(f'Epoch [{epoch+1}/{epochs}]\n', 
                  f'Loss: {running_loss/len(train_loader):.4f}\n',
                  f'Train Accuracy: {final_train_accuracy * 100:.2f}\n',
                  '--------------------------------------------------\n')

            final_test_accuracy = self.evaluate(test_loader=test_loader, test_accuracy=test_accuracy)
            if write_log:
                logging.info(f"Epoch: {epoch + 1}, Loss: {running_loss/len(train_loader):.4f}, Train accuracy: {final_train_accuracy}\t|\t Test accuracy: {final_test_accuracy}\n")

        print('======================Finished=========================')
        return running_loss / len(train_loader)

    def evaluate(self, test_loader=None, test_accuracy=None):
        test_loader = test_loader or self.dataloader.load_data(train=False)
        test_accuracy = test_accuracy or torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
        self.model.eval()
        test_accuracy.reset()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                test_accuracy.update(outputs.argmax(dim=1), labels)

        final_test_accuracy = test_accuracy.compute()
        print(f'Test Accuracy: {final_test_accuracy * 100:.2f}\n',
              '--------------------------------------------------\n')
        return final_test_accuracy

    def predict(self, data):
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for input in data:
                input = input.to(self.device)
                outputs = self.model(input)
                predictions.append(outputs.argmax(dim=1))

        return predictions

    def save(self, name=None):
        parent = 'models'
        checkpoint_path = name or 'mlp_checkpoint.pth'
        path = f'{parent}/{checkpoint_path}'
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'epochs': self.epochs
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")