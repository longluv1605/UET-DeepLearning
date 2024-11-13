import torchvision
from torch.utils.data import DataLoader

def get_dataloader(transform: torchvision.transforms,
                   batch_size: int,
                   num_workers: int):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    train_dataloader = DataLoader(dataset= trainset, batch_size = batch_size, shuffle= True, num_workers = num_workers)
    test_dataloader = DataLoader(dataset= testset, batch_size = batch_size, shuffle= False, num_workers = num_workers)

    return train_dataloader, test_dataloader