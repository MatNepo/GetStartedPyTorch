import torch
from torch import optim
from data_loading import load_data
from model import NeuralNetwork
from train import train
from test import test

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = NeuralNetwork()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
    print("Done!")

    test(test_loader, model, loss_fn)
