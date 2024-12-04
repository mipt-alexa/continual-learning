from torchvision import models
from classes import MyModel
from torch import optim, nn
import torch
import timm


# models_dict = {"resnet": models.resnet50(), 
#                "vit": timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=False),
#                "default": models.resnet50()}

optim_dict = {}

def create_model(model_name):
    if model_name == "vit": 
        return MyModel(timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=False),
                      upscale=True)
    else:
        return MyModel(models.resnet50())


def setup_optimizer(model_params, lr, weight_decay=0.):
    return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)


def setup_scheduler(optimizer, mode="", num_tasks=10, num_epochs_per_task=20):
    # if mode == "full":
        
    #     # scheduler_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1, total_iters=10)
    #     # scheduler_2 = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    #     # # scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #     # scheduler_3 = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    #     # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler_1, scheduler_2, scheduler_3], milestones=[10, 15])
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)

    # else:
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.6)

    if mode == "full":
        milestones = torch.tensor(range(1, 6), dtype=int) * 2 * num_epochs_per_task
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones.tolist(), gamma=0.2)
    elif mode == "const":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)
    elif mode == "cyclic":

        list_sched = []
        for _ in range(num_tasks):
            list_sched.append(optim.lr_scheduler.LinearLR(optimizer))
            # list_sched.append(optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95))
            list_sched.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, 1.5*num_epochs_per_task))


        milestones = [4, num_epochs_per_task]
        for _ in range(num_tasks-1):
            milestones.append(milestones[-1] + 4)
            milestones.append(milestones[-1] + num_epochs_per_task - 4)

        milestones = milestones[:-1]

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, list_sched, milestones=milestones)

    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    return scheduler
