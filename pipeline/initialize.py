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


def setup_optimizer(model_params, lr):
    return optim.Adam(model_params, lr=lr)

