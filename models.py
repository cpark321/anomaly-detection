import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

class MVTecSimpleCNN(nn.Module):
    def __init__(self):
        super(MVTecSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(256*5*5, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 256*5*5)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MVTecResNet(nn.Module):
    def __init__(self):
        super(MVTecResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)


        # if model_freeze:      --> model freeze: 성능 좋지 않았음. 정확도 5% 이상 감소
        #     for param in self.model.parameters():
        #         param.requires_grad = False


        # self.model.fc = Identity()   --> fc layer 결정을 위한 test -> 256x256 input 시 512
        self.model.fc = nn.Sequential(
            nn.Linear(512,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

