from mlp import MLP
from dataloader import Data
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torchmetrics


class Model:
    def __init__(
        self, model=None, criterion=None, optimizer=None, dataloader=None, epochs=100
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
        else:
            self.model = MLP(
                n_classes=10,
                n_hidden_nodes=100,
                image_width=32,
                image_height=32,
                color_channels=3,
            )
        self.model.to(self.device)

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = CrossEntropyLoss()

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = SGD(self.model.parameters(), lr=0.005)

        if dataloader is not None:
            self.dataloader = dataloader
        else:
            self.dataloader = Data(root="./data", batch_size=64, num_workers=4)

        self.epochs = epochs

    def training_and_evaluate(self):
        train_loader = self.dataloader.load_train_data()
        test_loader = self.dataloader.load_test_data()

        train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            self.device
        )
        test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            self.device
        )

        for epoch in range(self.epochs):
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

                _, predicted = torch.max(outputs.data, 1)
                train_accuracy.update(predicted, labels)

            self.model.eval()
            test_accuracy.reset()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_accuracy.update(predicted, labels)

            final_train_accuracy = train_accuracy.compute()
            final_test_accuracy = test_accuracy.compute()

            print(
                f"Epoch [{epoch+1}/{self.epochs}]\n",
                f"Loss: {running_loss/len(train_loader):.4f}\n",
                f"Train Accuracy: {final_train_accuracy:.2f}\n",
                f"Test Accuracy: {final_test_accuracy:.2f}\n",
                "--------------------------------------------------\n",
            )

        print("======================Finished=========================")

    def predict(self, data):
        prediction = []

        with torch.no_grad():
            for input in data:
                input = input.to(self.device)
                outputs = self.model(input)
                _, predicted = torch.max(outputs.data, 1)
                prediction.append(predicted)

        return prediction

    def save():
        pass
