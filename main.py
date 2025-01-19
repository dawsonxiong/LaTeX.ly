import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timm

# Pytorch Dataset

class LaTeXlyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

dataset = LaTeXlyDataset(
    data_dir = r'data'
)

data_dir = r'data'
target_to_class = {
    0: r'!', 1: r'0', 2: r'1', 3: r'2', 4: r'3', 5: r'4', 6: r'5', 7: r'6', 8: r'7', 9: r'8', 10: r'9',
    11: r'a', 12: r'|', 13: r'\alpha', 14: r'\approx', 15: r'b', 16: r'\beta', 17: r'c', 18: r'\cap',
    19: r'\cdot', 20: r',', 21: r'\cos', 22: r'\cot', 23: r'\csc', 24: r'\cup', 25: r'd', 26: r'-',
    27: r'\delta', 28: r'\div', 29: r'e', 30: r'\emptyset', 31: r'\epsilon', 32: r'=', 33: r'\equiv',
    34: r'\exists', 35: r'f', 36: r'\forall', 37: r'g', 38: r'\gamma', 39: r'\geq', 40: r'>', 41: r'h',
    42: r'i', 43: r'\in', 44: r'\infty', 45: r'\int', 46: r'j', 47: r'k', 48: r'l', 49: r'\lambda', 50: r'(',
    51: r'{', 52: r'\leftarrow', 53: r'\leq', 54: r'<', 55: r'm', 56: r'\mp', 57: r'\mu', 58: r'n', 59: r'\neq',
    60: r'\notin', 61: r'\nsubset', 62: r'\nsupset', 63: r'o', 64: r'\omega', 65: r'p', 66: r'\parallel',
    67: r'\partial', 68: r'\perp', 69: r'\phi', 70: r'\pi', 71: r'\plus', 72: r'\pm', 73: r'\propto',
    74: r'q', 75: r'r', 76: r') ', 77: r'}', 78: r'\rightarrow', 79: r's', 80: r'\sec', 81: r'\setminus',
    82: r'\sigma', 83: r'\sim', 84: r'\simeq', 85: r'\sin', 86: r'/', 87: r'\subset', 88: r'\subseteq',
    89: r'\supset', 90: r'\supseteq', 91: r't', 92: r'\tan', 93: r'\tau', 94: r'\therefore', 95: r'\theta',
    96: r'\times', 97: r'u', 98: r'v', 99: r'w', 100: r'x', 101: r'y', 102: r'z'
}

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])
dataset = LaTeXlyDataset(data_dir, transform)

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

for images, labels in dataloader:
    break

# Pytorch Model

class Latexly(nn.Module): 
    def __init__(self, num_classes = 103):
        super(Latexly, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
model = Latexly(num_classes = 103)

for image, label in dataset:
    break

example_out = model(images)
print(example_out.shape) # [batch_size, num_classes]

# Training loop

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
print(criterion(example_out, labels))

