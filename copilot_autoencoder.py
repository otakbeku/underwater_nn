import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class deep_autoencoder(nn.Module):
    def __init__(self):
        super(deep_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Linear(200, 50),
            nn.ReLU(True),
            nn.Linear(50, 10),
            nn.ReLU(True),
            nn.Linear(10, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(True),
            nn.Linear(10, 50),
            nn.ReLU(True),
            nn.Linear(50, 200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x