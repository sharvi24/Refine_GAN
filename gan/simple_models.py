import torch

import torch.nn as nn
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()        
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size = 4, stride = 1, padding = 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn1(self.conv2(x)))
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.leaky_relu(self.bn3(self.conv4(x)))
        x = self.conv5(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim=129, output_channels=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.conv1 = nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size = 4, stride = 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, output_channels, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x):
        return x
