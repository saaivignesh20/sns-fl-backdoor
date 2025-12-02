import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(config):
    """
    Download MNIST, split into per-client loaders, and return test loader.
    """
    data_root = Path(__file__).resolve().parent.parent / "data"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(data_root), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_root), train=False, download=True, transform=transform
    )

    num_clients = config.NUM_CLIENTS
    shard_size = len(train_dataset) // num_clients
    indices = torch.randperm(len(train_dataset))

    client_loaders = []
    for cid in range(num_clients):
        start = cid * shard_size
        end = (cid + 1) * shard_size if cid < num_clients - 1 else len(indices)
        shard_indices = indices[start:end]
        subset = Subset(train_dataset, shard_indices)
        loader = DataLoader(
            subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        client_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    return client_loaders, test_loader


def fedavg(state_dicts, weights):
    """
    Compute weighted average of model parameters.
    """
    total_weight = float(sum(weights))
    if total_weight == 0:
        raise ValueError("Total weight for FedAvg is zero.")

    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.zeros_like(state_dicts[0][key])
        for state, weight in zip(state_dicts, weights):
            avg_state[key] += state[key] * (float(weight) / total_weight)
    return avg_state


def aggregate_median(state_dicts):
    """
    Coordinate-wise median aggregation over client state_dicts.
    """
    if not state_dicts:
        raise ValueError("No state_dicts provided for median aggregation.")

    median_state = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([state[key] for state in state_dicts], dim=0)
        median_state[key] = torch.median(stacked, dim=0).values
    return median_state
