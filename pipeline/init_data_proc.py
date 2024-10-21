import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm


train_data = datasets.CIFAR100(
    root="/nfs/scistore14/chlgrp/avolkova/rotation1/data",
    train=True,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Resize(224), 
                            T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
                            T.RandomHorizontalFlip(), 
                            T.RandomRotation(10),
                            )
)

test_val_data = datasets.CIFAR100(
    root="/nfs/scistore14/chlgrp/avolkova/rotation1/data",
    train=False,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Resize(224))
    )

torch.save(train_data, '../data/train_data.pt')
torch.save(test_val_data, '../data/test_val_data.pt')
