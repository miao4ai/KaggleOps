import mlflow
import torch
import yaml
import os
import random
import numpy as np
from torchvision import models
from src.data_module import get_dataloaders
from src.model_module import AnimalClassifier
from src.utils import set_seed, get_device

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg['project']['seed'])
    device = get_device()

    # Setup MLflow
    if cfg['logging']['use_mlflow']:
        mlflow.set_experiment(cfg['project']['experiment_name'])
        mlflow.start_run(run_name="run_" + cfg['model']['name'])
        mlflow.log_params({
            "epochs": cfg['train']['epochs'],
            "lr": cfg['train']['lr'],
            "model": cfg['model']['name']
        })

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = AnimalClassifier(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    # Training loop
    for epoch in range(cfg['train']['epochs']):
        model.train_one_epoch(train_loader, optimizer, device)
        acc = model.evaluate(val_loader, device)
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

        if cfg['logging']['use_mlflow']:
            mlflow.log_metric("val_acc", acc, step=epoch)

    # Save model
    if cfg['logging']['use_mlflow']:
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    main(args.config)