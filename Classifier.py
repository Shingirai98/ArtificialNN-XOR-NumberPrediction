import copy
import os
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import requests ##
from PIL import Image
from io import BytesIO

import copy

batch = 64
T = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor()])
root = "./MNIST_JPGS/trainingSet/trainingSet"
data = ImageFolder(root, transform=T)

train_size = int(0.1*len(data))
val_size = len(data)-train_size

train_data, val_data = random_split(data, [train_size, val_size])

#val_data = ImageFolder('./MNSIT/trainingSet/trainingSet/', transform=T)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size = batch)

#plt.imshow(train_data[0][0][0], cmap='gray')


def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model


def validate_model(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to("cpu") # might skip this
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)

    return correct/total


def train(epochs = 3, learning_rate = 1e-3, device="cpu"):
    cnn = create_lenet().to(device)
    accuracies = []
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
    max_accuracy = 0

    for epoch in range(epochs):
        for i, (images, labels) in enumerate (train_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate_model(cnn, val_data_loader))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving best model with Accuracy: ", accuracy)
        print("Epoch:", epoch+1, "Accuracy: ",accuracy)
    plt.plot(accuracies)
    return best_model

lenet = train(5, "cpu")
