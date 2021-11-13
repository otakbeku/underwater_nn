# create a semi-supervised model
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class bert_model(torch.nn.Module):
    """Bert Model generated by copilot
    """
    def __init__(self, config):
        super(bert_model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_dir)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class Text_classificatio(torch.nn.Module):
    """Text classification using BERT
    """
    def __init__(self, config):
        super(Text_classificatio, self).__init__()
        self.bert = bert_model(config)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

def train_loader():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader

def validation_loader():
    validation_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    return validation_loader

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0