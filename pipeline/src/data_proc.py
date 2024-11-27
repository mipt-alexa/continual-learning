from collections import defaultdict
import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

from tqdm import tqdm

import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)



data_path = "/nfs/scistore23/chlgrp/avolkova/rotation1/pipeline/data/"



def create_loaders(buffer_size=0, num_tasks=10, batch_size=128):
    
    loaders = {key: [None for _ in range(num_tasks)] for key in ["train", "val", "test"]}

    # Val and test
    for name in ["val", "test"]:
        for i in tqdm(range(num_tasks)):
            data = torch.load(f"{data_path}/cil/{name}/{i}.pt")
                
            loaders[name][i] = DataLoader(data, batch_size=batch_size,
                                   shuffle=True, num_workers=8, pin_memory=True)

    # Train
    train_data = [None for _ in range(num_tasks)]


    label_to_indx = defaultdict(lambda: torch.empty(0, dtype=int), {})
    for i in tqdm(range(num_tasks)):
        train_data[i] = torch.load(f"{data_path}/cil/train/{i}.pt")
        for indx, (_, label) in enumerate(train_data[i]):
            label_to_indx[label] = torch.cat((label_to_indx[label], torch.tensor([indx], dtype=int)))


    buffer_label_to_indx = defaultdict(lambda: torch.empty(0, dtype=int), {})

    num_labels_per_task = 100 // num_tasks

    for task_id in tqdm(range(num_tasks)):
        
        subsets = [] # Subsets of differrent task to compose training data with replay

        if task_id > 0:

            buffer_per_label = buffer_size // task_id // num_labels_per_task

            # Reduce the number of old samples in replay buffer
            for label in range(num_labels_per_task * (task_id - 1)):
                perm = torch.randperm(len(buffer_label_to_indx[label]))
                croppped_ind = perm[:buffer_per_label]

                buffer_label_to_indx[label] = buffer_label_to_indx[label][croppped_ind]

            # Add to replay buffr samples from the previous task
            for label in range(num_labels_per_task * (task_id - 1), num_labels_per_task * task_id):
                perm = torch.randperm(len(label_to_indx[label]))
                croppped_ind = perm[:buffer_per_label]

                buffer_label_to_indx[label] = label_to_indx[label][croppped_ind]

            buffer_task_to_indx = defaultdict(lambda: torch.empty(0, dtype=int), {})

            # Convert to indices within task 
            for label, indx in buffer_label_to_indx.items():
                task = label // num_labels_per_task
                buffer_task_to_indx[task] = torch.cat((buffer_task_to_indx[task], indx))

            # Store Subsets for each task
            for task, indx in buffer_task_to_indx.items():
                subsets.append(Subset(train_data[task], indx))
            
        # Add the new task
        subsets.append(train_data[task_id])  
        data_with_replay = ConcatDataset(subsets)

        loaders["train"][task_id] = DataLoader(data_with_replay, batch_size=batch_size,
                                                shuffle=True, num_workers=8, pin_memory=True)

    return loaders
    


def create_full_ds_loaders(batch_size=128):
    
    train_data = datasets.CIFAR100(
    root=data_path,
    train=True,
    download=False,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                            T.RandomHorizontalFlip(), 
                            T.RandomCrop((32, 32), padding=6, padding_mode="symmetric"),
                            # T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
                            )
    )

    test_val_data = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=False,
        transform=nn.Sequential(T.ToImage(), 
                                T.ToDtype(torch.float32, scale=True), 
                                T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
                               )
    )
    
    
    train_data, val_data = random_split(train_data, [45000, 5000])
    data_dict = {"train": train_data, "val": val_data}
    
    loaders_dict = {}
    
    for name, data in data_dict.items():
        loaders_dict[name] = [DataLoader(data, batch_size=batch_size,
                                   shuffle=True, num_workers=8)]
    
    return loaders_dict
