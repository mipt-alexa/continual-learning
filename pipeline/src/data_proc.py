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
    replay_samples = [None for _ in range(num_tasks)]

    labels_per_task = 100 // num_tasks
    target_sep = [(i*labels_per_task, (i+1)*labels_per_task) for i in range(num_tasks)]
    target_sep[-1] = (target_sep[-1][0], 100)

    # First load all training data
    for i in range(num_tasks):
        train_data[i] = torch.load(f"{data_path}/cil/train/{i}.pt")

        label_to_indices = defaultdict(lambda: torch.empty(0, dtype=int), {})
        for idx, (_, label) in enumerate(train_data[i]):
            label_to_indices[label] = torch.cat((label_to_indices[label], torch.tensor([idx], dtype=int)))

        
        if buffer_size > 0:
            # Get labels for current task
            
            num_labels = len(label_to_indices.keys())
            
            # Calculate samples per class
            samples_per_task = buffer_size // (i + 1) 
            samples_per_label = samples_per_task // num_labels
            
            # Select equal samples for each class
            task_replay_indices = []

            for label, indx in label_to_indices.items():
                perm = torch.randperm(len(indx))
                selected_indices = indx[perm[:samples_per_label]]
                task_replay_indices.append(selected_indices)
            
            # Combine all selected indices
            replay_samples[i] = torch.cat(task_replay_indices)

    for task_id in tqdm(range(num_tasks)):
        if task_id > 0 and buffer_size > 0:
            # Create replay buffer from fixed samples of previous tasks
            replay_buffer = []
            
            samples_per_task = buffer_size // (i + 1) 

            for prev_task_id in range(task_id):
                perm = torch.randperm(len(replay_samples[prev_task_id]))
                replay_buffer.append(Subset(train_data[prev_task_id], replay_samples[prev_task_id][perm[:samples_per_task]]))
            
            # Combine current task data with replay buffer
            data = ConcatDataset([train_data[task_id]] + replay_buffer)
        else:
            data = train_data[0]
            
        loaders["train"][task_id] = DataLoader(data, batch_size=batch_size,
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
