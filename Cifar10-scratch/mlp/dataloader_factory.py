import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class DataLoaderFactory:
    def __init__(self, root='./data', transform=None, batch_size=32, num_workers=2, download=True):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

    def load_data(self, train=True):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, 
            train=train, 
            download=self.download, 
            transform=self.transform
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=train, 
            num_workers=self.num_workers
        )