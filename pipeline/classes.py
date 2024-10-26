import torch
from torch import nn
from train import train, train_class_incremental


class MyModel(nn.Module):
    def __init__(self, backbone):
        super(MyModel, self).__init__()
        self.backbone = backbone
        self.name = type(backbone).__name__
        
        self.top_layers = nn.Sequential(nn.ReLU(), 
                                        nn.Linear(1000, 100),
                                        nn.LogSoftmax(dim=1))
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.top_layers(x)
        return x


class ExperimentTrainer():
    def __init__(self, dataloaders, model, optimizer, loss_fn, device, num_tasks=10):
        self.device = device
        self.loaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss_fn

        self.num_tasks = num_tasks
        self.acc_full = [[] * num_tasks]
        self.acc_last = []
        self.acc_0 = []

    def train_full(self, task_id=0, num_epochs=20):
        loss, acc_train, acc_val = train(self.model, 
                                         self.device, 
                                         self.optimizer, 
                                         self.criterion, 
                                         self.loaders["train"][task_id], 
                                         self.loaders["val"][task_id], 
                                         num_epochs)
        self.acc_full[task_id] = acc_val
        print("Model: ", self.model.name)
        print("Accuracy on task", task_id, ": ", acc_val)
        torch.save(acc_val, f"./results/{self.model}_{num_epochs}_{self.optimizer.lr}.pt")
        # visualize(loss, (acc_train, acc_val), filename="_".join(["task", str(task_id), self.model.name, str(num_epochs), "ep"]))
        

    def train_class_inc(self, num_epochs_per_task=10):
        acc_last, acc_0 = train_class_incremental(self.model, self.device, self.optimizer, 
                                                    self.criterion, self.num_tasks, 
                                                    num_epochs_per_task)
        self.acc_last = acc_last
        self.acc_0 = acc_0

        print("Model: ", self.model.name )
        # visualize_class_incremental(self.acc_last, self.acc_0, 
        #                             filename="_".join(["CIL", str(self.num_tasks), self.model.name, str(num_epochs_per_task), "ep"])
                                   # )
