import torch
import torch.nn as nn
from torchvision import models

class AnimalClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base = models.__dict__[cfg['model']['name']](pretrained=cfg['model']['pretrained'])
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, cfg['model']['num_classes'])
        self.model = base

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, dataloader, optimizer, device):
        self.train()
        criterion = nn.CrossEntropyLoss()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = self(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    def evaluate(self, dataloader, device):
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                preds = self(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total