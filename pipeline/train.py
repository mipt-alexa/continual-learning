import torch
from torch import optim, nn

from torchvision import datasets, models
import torchvision.transforms.v2 as T

from tqdm import tqdm


def evaluate(model, device, test_loader):
    model.eval()
    scores = torch.zeros(1).to(device)
    for data, target in test_loader:
        pred = model(data.to(device)).argmax(dim=1)
        scores += (pred == target.to(device)).float().mean(dim=0)
    return scores.item() / len(test_loader) 


def train(model, device, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=30):
    losses = torch.zeros(num_epochs).to(device)
    acc_scores = torch.empty((2, num_epochs)).to(device) 
    
    for i in tqdm(range(num_epochs)):
        model.train()
        
        for data, target in train_loader:    
            optimizer.zero_grad()
            output = model(data.to(device))            
            loss = criterion(output, target.to(device))
            losses[i] += loss.data.to("cpu") / len(train_loader)

            loss.backward()
            optimizer.step()
            
        scheduler.step()

        model.eval()
        acc_scores[0][i] = evaluate(model, device, train_loader)
        acc_scores[1][i] = evaluate(model, device, val_loader)
           
        print(f"Epoch {i}: loss={loss.item()}, train_accuracy={acc_scores[0][i]}, val_accuracy={acc_scores[1][i]}")

    return losses, acc_scores[0], acc_scores[1]



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

