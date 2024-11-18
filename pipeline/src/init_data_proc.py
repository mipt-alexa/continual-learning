import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm



data_path = "/nfs/scistore23/chlgrp/avolkova/rotation1/pipeline/data"


def save_data_cil(dataset, name, target_sep, replay=False):
    indx = torch.empty(0, dtype=torch.int64)
    
    for i, separator in tqdm(enumerate(target_sep)):
        current_indx = torch.tensor([j for j, (_, target) in enumerate(dataset) if separator[0] <= target < separator[1]], dtype=torch.int64)
                
        if replay and name == "train":
            indx = torch.cat((indx, current_indx))
        else:
            indx = current_indx
            
        subset = Subset(dataset, indx)

        if replay and name == "train":
            torch.save(subset, f"{data_path}/cil_replay/{name}/{i}.pt")
        else:
            torch.save(subset, f"{data_path}/cil/{name}/{i}.pt")

    print(f"{name} dataset for class incremental learning saved (reply={replay})")


train_data = datasets.CIFAR100(
    root=data_path,
    train=True,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            # T.Resize(224), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                            T.RandomHorizontalFlip(), 
                            T.RandomCrop((32, 32), padding=6, padding_mode="symmetric"),
                            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
                            )
)

test_val_data = datasets.CIFAR100(
    root=data_path,
    train=False,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            # T.Resize(224),
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
                           )
)


val_data, test_data = random_split(test_val_data, [5000, 5000])
data_dict = {"train": train_data, "val": val_data, "test": test_data}


num_tasks = 10
set_size = 100 // num_tasks

target_sep = [(i*set_size, (i+1)*set_size) for i in range(num_tasks)]
target_sep[-1] = (target_sep[-1][0], 100)

for name, dataset in data_dict.items():
    save_data_cil(dataset, name, target_sep, replay=False)
    if name == "train":
        save_data_cil(dataset, name, target_sep, replay=True)

