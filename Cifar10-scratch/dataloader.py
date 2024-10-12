import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class Data():
    def __init__(self, root='./data', transform=None, batch_size=64, num_workers=2):
        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def load_train_data(self):
        train_dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=self.transform)

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def load_test_data(self):
        test_dataset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.transform)

        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)    