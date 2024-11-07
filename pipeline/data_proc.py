import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm

import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


train_data = torch.load('../data/train_data.pt')
val_data = torch.load('../data/val_data.pt')
test_data = torch.load('../data/test_data.pt')

data_dict = {"train": train_data, "val": val_data, "test": test_data}


def get_image_shape():
    return (1,) + tuple(train_data[0][0].shape)


def create_loaders_from_subset(dataset, name, num_classes, batch_size=128):
    loaders = [None for _ in range(num_classes)]
    
    for i in tqdm(range(num_classes)):
        indx = torch.load(f"/nfs/scistore23/chlgrp/avolkova/rotation1/data/{name}_indx/{i}.pt")
        subset = Subset(dataset, indx)
        loaders[i] = DataLoader(subset, batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    return loaders


def create_loaders(num_tasks = 10):
    loaders_dict = {}
    for k,v in data_dict.items():
        loaders_dict[k] = create_loaders_from_subset(v, k, num_tasks)
        
    return loaders_dict


def create_full_ds_loaders(batch_size=64):
    loaders_dict = {}
    
    for name, data in data_dict.items():
        loaders_dict[name] = [DataLoader(data, batch_size=batch_size,
                                   shuffle=True, num_workers=4)]
    
    return loaders_dict
    
