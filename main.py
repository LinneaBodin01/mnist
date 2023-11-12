import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from network.network import SimpleNN

def main():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./mnist_data", train=True, transform=transform, download=True
    )

    eval_dataset = datasets.MNIST(
        root="./mnist_data", train=False, transform=transform, download=True
    )

    model = SimpleNN(784, 100, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False)

    for _ in range(10):
        for images, labels in train_loader:
            outputs = model(images.view(-1, 784))
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                outputs = model(images.view(-1, 784))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        print("Accuracy: {}".format(100 * correct / total))

    #torch.save(model.state_dict(), "./model.pth")

    


if __name__ == "__main__":
    main()