import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm

import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)



data_path = "/nfs/scistore23/chlgrp/avolkova/rotation1/pipeline/data/"


def get_image_shape():
    return (1,) + tuple(train_data[0][0].shape)


data_path = "/nfs/scistore23/chlgrp/avolkova/rotation1/pipeline/data"

def create_loaders(replay=False, num_tasks=10, batch_size=64):
    
    loaders = {key: [None for _ in range(num_tasks)] for key in ["train", "val", "test"]}

    for name in loaders.keys():
        for i in tqdm(range(num_tasks)):
            if replay and name == "train":
                data = torch.load(f"{data_path}/cil_replay/{name}/{i}.pt")
            else:
                data = torch.load(f"{data_path}/cil/{name}/{i}.pt")
                
            loaders[name][i] = DataLoader(data, batch_size=batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)

    return loaders


def create_full_ds_loaders(batch_size=128):
    print("Not implemented")
    train_data = torch.load(data_path + "train_data.pt")
    val_data = torch.load(data_path + "val_data.pt")
    test_data = torch.load(data_path + "test_data.pt")
    
    data_dict = {"train": train_data, "val": val_data, "test": test_data}
    
    loaders_dict = {}
    
    for name, data in data_dict.items():
        loaders_dict[name] = [DataLoader(data, batch_size=batch_size,
                                   shuffle=True, num_workers=8)]
    
    return loaders_dict
