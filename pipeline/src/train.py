import torch
from torch import optim, nn

from torchvision import datasets, models
import torchvision.transforms.v2 as T

from tqdm import tqdm
import wandb


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The epoch parameter in `scheduler.step()` was not necessary and is being deprecated")


def evaluate(model, device, loader):
    model.eval()
    scores = torch.zeros(1).to(device)
    for data, target in loader:
        pred = model(data.to(device)).argmax(dim=1)
        scores += (pred == target.to(device)).float().mean(dim=0)
    return scores.to(device) / len(loader) 


def evaluate_all_tasks(model, device, loaders, num_tasks=10):
    acc = torch.empty(0).to(device)
    model.eval()

    for t in range(num_tasks):
        acc = torch.cat((acc, evaluate(model, device, loaders[t])))
    return acc


def train(model, device, optimizer, scheduler, criterion, train_loaders, val_loaders, task_id, num_epochs=30, num_tasks=10):
    losses = torch.zeros(num_epochs).to(device)
    # acc_scores = torch.empty((2, num_epochs)).to(device) 

    train_acc = torch.empty((0, num_tasks)).to(device)
    val_acc = torch.empty((0, num_tasks)).to(device)
    
    for i in tqdm(range(num_epochs)):
        model.train()
        
        for data, target in train_loaders[task_id]:    
            optimizer.zero_grad()
            output = model(data.to(device))            
            loss = criterion(output, target.to(device))
            losses[i] += loss.data.to("cpu") / len(train_loaders[task_id])

            loss.backward()
            optimizer.step()
            
        scheduler.step()

        last_train_acc = evaluate_all_tasks(model, device, train_loaders, num_tasks)
        train_acc = torch.cat((train_acc, last_train_acc.unsqueeze(0)))
        
        last_val_acc = evaluate_all_tasks(model, device, val_loaders, num_tasks)
        val_acc = torch.cat((val_acc, last_val_acc.unsqueeze(0)))

        wandb.log({"loss": losses[i].item(), 
                   "lr": optimizer.param_groups[0]["lr"], 
                   **{f"train_acc_{j}": value for j, value in enumerate(last_train_acc.tolist())}, 
                   **{f"val_acc_{j}": value for j, value in enumerate(last_val_acc.tolist())}})
                                     
        print(f"Epoch {i}: loss = {losses[i].item():5.2f}, train_acc = {train_acc[-1][task_id]:.2f}, val_acc = {val_acc[-1][task_id]:.2f}")

    return losses, train_acc, val_acc



# def train_class_incremental(model, device, optimizer, scheduler, criterion, num_tasks, num_epochs_per_task):
#     acc_on_task_0 = torch.empty(num_tasks)
#     acc_on_last_task = torch.empty(num_tasks)
    
#     for i in range(num_tasks):
#         _, acc_train, acc_val = train(model, optimizer, scheduler, criterion, train_loaders[i], val_loaders[i], num_epochs_per_task)

#         acc_on_last_task[i] = acc_val[-1]
#         acc_on_task_0[i] = evaluate(model, device, val_loaders[0])

#         print("Task ", i)
#         print("Accuracy on task ", i, ": ", acc_on_last_task[i], "Accuracy on task ", 0, ": ",acc_on_task_0[i])
        
#     return acc_on_last_task, acc_on_task_0


def zero_out_hook(grad, cond):
    grad[:cond[0]] = 0
    grad[cond[1]:] = 0
    return grad


# def train_class_inc_hook(model, device, optimizer, scheduler, criterion, num_tasks, num_epochs_per_task, hook):
#     acc_on_task_0 = torch.empty(num_tasks)
#     acc_on_last_task = torch.empty(num_tasks)

#     set_size = 100 // num_tasks
#     target_sep = [(i*set_size, (i+1)*set_size) for i in range(num_tasks)]
#     target_sep[-1] = (target_sep[-1][0], 100)
    
#     for i in range(num_tasks):
#         # register hooks
#         hook_w = model.top_layers[1].weight.register_hook(partial(zero_out_hook, 
#                                                                   cond=target_sep[i]))
#         hook_b = model.top_layers[1].bias.register_hook(partial(zero_out_hook, 
#                                                                 cond=target_sep[i]))

#         _, acc_train, acc_val = train(model, optimizer, scheduler, criterion, 
#                                       train_loaders[i], val_loaders[i], num_epochs_per_task)

#         # remove the hooks
#         hook_w.remove()
#         hook_b.remove()
        
#         acc_on_last_task[i] = acc_val[-1]
#         acc_on_task_0[i] = evaluate(model, device, val_loaders[0])

#         print("Task ", i)
#         print("Accuracy on task ", i, ": ", acc_on_last_task[i], "Accuracy on task ", 0, ": ",acc_on_task_0[i])
        
#     return acc_on_last_task, acc_on_task_0

