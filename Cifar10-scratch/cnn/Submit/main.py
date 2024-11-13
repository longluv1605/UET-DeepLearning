import torch
import torchvision # type: ignore
from torchvision import transforms # type: ignore
from torch import nn # type: ignore
import os

from cnn import CNN
from dataloader import get_dataloader
from model_controller import *
from utils import *
from pretrained import *

def use_pretrained_model(model_name, class_names):
    model, model_transform = create_model(name=model_name, class_names=class_names)
    summarize_model(model)
    
    model_train_loader, model_test_loader = get_dataloader(model_transform, batch_size= 32, num_workers=2)
    model_lossfn, model_optimizer = initialize_lossfn_and_optimizer(model=model)
    train_model(trainloader = model_train_loader,
            testloader = model_test_loader,
            model= model,
            loss_fn = model_lossfn,
            optimizer= model_optimizer,
            epochs= 5)
    
    save_model(model=model, name=model_name + '_based')

def main():
    print("PyTorch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    class_names = cifar10.classes
    print(class_names)
    
    num_workers = os.cpu_count()

    simple_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    simple_cnn_train_dataloader, simple_cnn_test_dataloader = get_dataloader(simple_transforms, 32, num_workers)
    
    cnn = CNN(3, 10, 10).to(device)
    simple_cnn_loss_fn, simple_cnn_optimizer = initialize_lossfn_and_optimizer(model = cnn)

    train_model(trainloader = simple_cnn_train_dataloader,
            testloader = simple_cnn_test_dataloader,
            model= cnn,
            loss_fn = simple_cnn_loss_fn,
            optimizer= simple_cnn_optimizer,
            epochs= 3,
            device= device,
            seed = 42, write_log='simple_cnn')
    
    save_model(model=cnn, name='simple_cnn')
    
    
if __name__ == '__main__':
    main()