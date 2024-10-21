import torch
from torch import optim, nn

from torchvision import datasets, models
import torchvision.transforms.v2 as T

from tqdm import tqdm


def evaluate(model, test_loader):
    model.eval()
    scores = 0.
    for data, target in test_loader:
        pred = model(data.to(device)).argmax(dim=1)
        scores += (pred == target.to(device)).float().mean(dim=0)
    return scores / len(test_loader) 


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs=30):
    losses = torch.zeros(num_epochs)
    acc_scores = torch.empty((2, num_epochs)) 
    
    for i in tqdm(range(num_epochs)):
        model.train()
        
        for data, target in train_loader:    
            optimizer.zero_grad()
            output = model(data.to(device))            
            loss = criterion(output, target.to(device))
            losses[i] += loss.data.to("cpu") / len(train_loader)

            loss.backward()
            optimizer.step()

        model.eval()
        acc_scores[0][i] = evaluate(model, train_loader)
        acc_scores[1][i] = evaluate(model, val_loader)

    return losses, acc_scores[0], acc_scores[1]


def train_class_incremental(model, optimizer, criterion, num_tasks, num_epochs_per_task):
    acc_on_task_0 = torch.empty(num_tasks)
    acc_on_last_task = torch.empty(num_tasks)
    
    for i in range(num_tasks):
        _, acc_train, acc_val = train(model, optimizer, criterion, train_loaders[i], val_loaders[i], num_epochs_per_task)

        acc_on_last_task[i] = acc_val[-1]
        acc_on_task_0[i] = evaluate(model, val_loaders[0])

        print("Task ", i)
        print("Accuracy on task ", i, ": ", acc_on_last_task[i], "Accuracy on task ", 0, ": ",acc_on_task_0[i])
        
    return acc_on_last_task, acc_on_task_0


