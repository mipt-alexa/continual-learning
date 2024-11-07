import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm


train_data = datasets.CIFAR100(
    root="/nfs/scistore23/chlgrp/avolkova/rotation1/data",
    train=True,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Resize(224), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                            T.RandomHorizontalFlip(), 
                            T.RandomCrop((224, 224), padding=20, padding_mode="symmetric"),
                            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
                            )
)

test_val_data = datasets.CIFAR100(
    root="/nfs/scistore23/chlgrp/avolkova/rotation1/data",
    train=False,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Resize(224),
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
                           )
)

torch.save(train_data, '../data/train_data.pt')
print("Train data saved")

val_data, test_data = random_split(test_val_data, [5000, 5000])
torch.save(val_data, '../data/val_data.pt')
print("Val data saved")
torch.save(test_data, '../data/test_data.pt')
print("Test data saved")


def indx_by_task(dataset, name, target_sep, batch_size=32):    
    for i, separator in tqdm(enumerate(target_sep)):
        indx = [j for j, (_, target) in enumerate(dataset) if separator[0] <= target < separator[1]]

        torch.save(torch.tensor(indx), f"/nfs/scistore23/chlgrp/avolkova/rotation1/data/{name}_indx/{i}.pt")
    print(name, "indices saved")


num_tasks = 10
set_size = 100 // num_tasks

target_sep = [(i*set_size, (i+1)*set_size) for i in range(num_tasks)]

indx_by_task(train_data, "train", target_sep)
indx_by_task(val_data, "val", target_sep)
indx_by_task(test_data, "test", target_sep)

