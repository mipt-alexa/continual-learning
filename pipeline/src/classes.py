import torch
from torch import nn
from train import train, evaluate, zero_out_hook

from functools import partial


class MyModel(nn.Module):
    def __init__(self, backbone, upscale=False):
        super(MyModel, self).__init__()

        if upscale:
            self.upscale_layer = nn.Upsample((34, 34))
        else:
            self.upscale_layer = nn.Identity()
            
        self.backbone = backbone
        self.name = type(backbone).__name__
        
        self.top_layers = nn.Sequential(nn.ReLU(), 
                                        nn.Linear(1000, 100),
                                        nn.LogSoftmax(dim=1))
    
    def forward(self, x):
        
        x = self.upscale_layer(x)
        x = self.backbone(x)
        x = self.top_layers(x)
        
        return x


class ExperimentTrainer():
    def __init__(self, dataloaders, model, optimizer, scheduler, loss_fn, device, num_epochs, num_tasks=10):
        self.device = device
        self.loaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = loss_fn
        self.num_epochs = num_epochs  # FIX uniform usage of number of epochs across different modes 

        self.num_tasks = num_tasks
        # self.acc_full = [[] * num_tasks]
        # self.acc_last = []
        # self.acc_0 = []

        self.losses = torch.empty(0).to(device)
        self.val_acc = torch.empty((0, num_tasks)).to(device)
        self.train_acc = torch.empty((0, num_tasks)).to(device)
        

    def train_full(self, task_id=0, num_epochs=20):
        loss, acc_train, acc_val = train(self.model, 
                                         self.device, 
                                         self.optimizer, 
                                         self.scheduler,
                                         self.criterion, 
                                         self.loaders["train"][task_id], 
                                         self.loaders["val"][task_id], 
                                         self.num_epochs)
        self.acc_full[task_id] = acc_val
        print(f"Accuracy on task {task_id}: {acc_val:.3f}")
        
        return loss, acc_train, acc_val
        

    def train_class_inc(self, num_epochs_per_task=10):
        
        for task_id in range(self.num_tasks):
            loss, train_acc, val_acc = train(self.model, 
                                         self.device, 
                                         self.optimizer, 
                                         self.scheduler,
                                         self.criterion, 
                                         self.loaders["train"], 
                                         self.loaders["val"], 
                                         task_id,
                                         num_epochs_per_task)

            self.val_acc = torch.cat((self.val_acc, val_acc))
            self.train_acc = torch.cat((self.train_acc, train_acc))
            self.losses = torch.cat((self.losses, loss))

            print(f"Finished training on task {task_id}... val_accuracy on task 0 = {val_acc[-1][0]:.3f}")
                         
        return self.losses, self.train_acc, self.val_acc

    
    def train_class_inc_hook(self, num_epochs_per_task=10):

        set_size = 100 // self.num_tasks
        target_sep = [(i*set_size, (i+1)*set_size) for i in range(self.num_tasks)]
        target_sep[-1] = (target_sep[-1][0], 100)

        for task_id in range(self.num_tasks):
            # register hooks
            hook_w = self.model.top_layers[1].weight.register_hook(partial(zero_out_hook, 
                                                                      cond=target_sep[task_id]))
            hook_b = self.model.top_layers[1].bias.register_hook(partial(zero_out_hook, 
                                                                cond=target_sep[task_id]))

            loss, train_acc, val_acc = train(self.model, 
                                         self.device, 
                                         self.optimizer, 
                                         self.scheduler,
                                         self.criterion, 
                                         self.loaders["train"], 
                                         self.loaders["val"], 
                                         task_id,
                                         num_epochs_per_task)

            # check is the hooks are working as planned
            weight = self.model.top_layers[1].weight
            print("weight", [a_chunk.norm().item() for a_chunk in weight.mean(axis=-1).split(10)])
            print("weight_grad", [a_chunk.norm().item() for a_chunk in weight.grad.mean(axis=-1).split(10)])

            bias = self.model.top_layers[1].bias
            print("bias", [a_chunk.norm().item() for a_chunk in bias.split(10)])            
            print("bias_grad", [a_chunk.norm().item() for a_chunk in bias.grad.split(10)])
            
            # remove the hooks
            hook_w.remove()
            hook_b.remove()

            self.val_acc = torch.cat((self.val_acc, val_acc))
            self.train_acc = torch.cat((self.train_acc, train_acc))
            self.losses = torch.cat((self.losses, loss))

            print(f"Finished training on task {task_id}... val_accuracy on task 0 = {val_acc[-1][0]:.3f}")
            
        return self.losses, self.train_acc, self.val_acc


    # def train_class_inc_replay(self, replay_epochs=1, num_epochs_per_task=10):
    #     # acc_on_task_0 = torch.empty(self.num_tasks).to(self.device)
    #     # acc_on_last_task = torch.empty(0).to(self.device)
    #     losses = torch.empty(0).to(self.device)
        
    #     for i in range(self.num_tasks):
    #         loss, acc_train, acc_val = train(self.model, 
    #                                      self.device, 
    #                                      self.optimizer, 
    #                                      self.scheduler,
    #                                      self.criterion, 
    #                                      self.loaders["train"][i], 
    #                                      self.loaders["val"][i], 
    #                                      num_epochs_per_task)

    #         self.val_acc = torch.cat((self.val_acc, val_acc))
    #         self.train_acc = torch.cat((self.train_acc, train_acc))
    #         losses = torch.cat((losses, loss))

    #         # add replay: extra training epochs for each previous tasks
    #         for replay_index in range(0, i):
    #             _, _, _ = train(self.model, 
    #                                              self.device, 
    #                                              self.optimizer, 
    #                                              self.scheduler,
    #                                              self.criterion, 
    #                                              self.loaders["train"][replay_index], 
    #                                              self.loaders["val"][replay_index], 
    #                                              num_epochs=replay_epochs)

    #         acc_on_task_0[i] = evaluate(self.model, self.device, self.loaders["val"][0])
            
    #         print(f"Finished training on task {i}... val_accuracy on task 0 = {acc_on_task_0[i]:.3f}")
                         
    #     return losses, acc_on_last_task, acc_on_task_0        
        

    