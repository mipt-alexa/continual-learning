from torchvision import models
from classes import MyModel
from torch import optim, nn
import timm


# models_dict = {"resnet": models.resnet50(), 
#                "vit": timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=False),
#                "default": models.resnet50()}

optim_dict = {}

def create_model(model_name):
    if model_name == "vit": 
        return MyModel(timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=False))
    else:
        return MyModel(models.resnet50())


def setup_optimizer(model_params, lr, weight_decay=0.):
    return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)


def setup_scheduler(optimizer, mode="", gamma=0.98, num_epochs=60):
    if mode == "full":
        
        scheduler_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1, total_iters=10)
        scheduler_2 = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        # scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler_3 = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler_1, scheduler_2, scheduler_3], milestones=[10, 15])

    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.)

    return scheduler
