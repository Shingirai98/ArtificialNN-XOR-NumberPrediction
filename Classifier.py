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

