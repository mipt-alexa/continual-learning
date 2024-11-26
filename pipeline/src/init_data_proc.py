import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import Subset
from collections import defaultdict

from tqdm import tqdm



data_path = "/nfs/scistore23/chlgrp/avolkova/rotation1/pipeline/data"


# def save_data_cil(dataset, name, target_sep, replay=False):
#     indx = torch.empty(0, dtype=torch.int64)
    
#     for i, separator in tqdm(enumerate(target_sep)):
#         current_indx = torch.tensor([j for j, (_, target) in enumerate(dataset) if separator[0] <= target < separator[1]], dtype=torch.int64)
                
#         if replay and name == "train":
#             indx = torch.cat((indx, current_indx))
#         else:
#             indx = current_indx
            
#         subset = Subset(dataset, indx)

#         if name == "train":
#             torch.save(subset, f"{data_path}/cil_replay/{name}/{i}.pt")
#         else:
#             torch.save(subset, f"{data_path}/cil/{name}/{i}.pt")

#     print(f"{name} dataset for class incremental learning saved (reply={replay})")


train_data = datasets.CIFAR100(
    root=data_path,
    train=True,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                            T.RandomHorizontalFlip(), 
                            T.RandomCrop((32, 32), padding=6, padding_mode="symmetric"),
                            # T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
                            )
)

test_data = datasets.CIFAR100(
    root=data_path,
    train=False,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
                           )
)


num_tasks = 10
labels_per_task = 100 // num_tasks

## Divide train_data into train and validation sets, such that each label would have an equal number of samples in both
## Start by mapping labels to corresponding indices in train_data 

label_to_indices = defaultdict(lambda: torch.empty(0, dtype=int), {})
for idx, (_, label) in enumerate(train_data):
    label_to_indices[label] = torch.cat((label_to_indices[label], torch.tensor([idx], dtype=int)))

## Define the number of validation samples per label (10% val, 90% train in this case)
val_per_label = len(train_data) // 10 // 100 

label_to_indx_val = defaultdict(lambda: torch.empty(0, dtype=int), {})
label_to_indx_train = defaultdict(lambda: torch.empty(0, dtype=int), {})

## Randomly choose indices for validation and training samples

for label, indx in label_to_indices.items():
    perm = torch.randperm(len(indx))
    label_to_indx_val[label] = torch.cat((label_to_indx_val[label], indx[perm[:val_per_label]]))
    label_to_indx_train[label] = torch.cat((label_to_indx_train[label], indx[perm[val_per_label:]]))

## Join indices for each task

task_to_indx_val = defaultdict(lambda: torch.empty(0, dtype=int), {})
task_to_indx_train = defaultdict(lambda: torch.empty(0, dtype=int), {})

for label, ind in label_to_indx_val.items():
    task_id = label // labels_per_task
    task_to_indx_val[task_id] = torch.cat((task_to_indx_val[task_id], ind))

for label, ind in label_to_indx_train.items():
    task_id = label // labels_per_task
    task_to_indx_train[task_id] = torch.cat((task_to_indx_train[task_id], ind))

## Save datasets for each task
for task_id, ind in task_to_indx_val.items():
    subset = Subset(train_data, ind)
    torch.save(subset, f"{data_path}/cil/val/{task_id}.pt")
print("Val data saved")

for task_id, ind in task_to_indx_train.items():
    subset = Subset(train_data, ind)
    torch.save(subset, f"{data_path}/cil/train/{task_id}.pt")
print("Train data saved")


## Save test dataset by task

target_sep = [(i*labels_per_task, (i+1)*labels_per_task) for i in range(num_tasks)]
target_sep[-1] = (target_sep[-1][0], 100)

for task_id, separator in tqdm(enumerate(target_sep)):
    current_indx = torch.tensor([j for j, (_, target) in enumerate(test_data) if separator[0] <= target < separator[1]], dtype=torch.int64)
            
    subset = Subset(test_data, current_indx)
    torch.save(subset, f"{data_path}/cil/test/{task_id}.pt")
    
print("Test data saved")
