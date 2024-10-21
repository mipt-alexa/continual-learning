import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split

from tqdm import tqdm


train_data = datasets.CIFAR100(
    root="data",
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
    root="data",
    train=False,
    download=True,
    transform=nn.Sequential(T.ToImage(), 
                            T.ToDtype(torch.float32, scale=True), 
                            T.Resize(224))
    )

val_data, test_data = random_split(test_val_data, [5000, 5000])


def get_image_shape():
    return (1,) + tuple(train_data[0][0].shape)


def split_by_task(dataset, target_sep, batch_size=32):
    loaders = [None for _ in range(len(target_sep))]
    
    for i, separator in tqdm(enumerate(target_sep)):
        indx = [j for j, (_, target) in enumerate(dataset) if separator[0] <= target < separator[1]]
        subset = Subset(dataset, indx)
        loaders[i] = DataLoader(subset, batch_size=batch_size,
                                   shuffle=True, num_workers=2)
    return loaders


def create_loaders(num_tasks = 10):
    set_size = 100 // num_tasks
    
    target_sep = [(i*set_size, (i+1)*set_size) for i in range(num_tasks)]
    
    train_loaders = split_by_task(train_data, target_sep)
    val_loaders = split_by_task(val_data, target_sep)
    test_loaders = split_by_task(test_data, target_sep)
    dict = {"train": train_loaders, "val": val_loaders, "test": test_loaders}
    return dict
