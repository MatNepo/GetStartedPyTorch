import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader
