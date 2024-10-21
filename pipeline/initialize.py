from torchvision import models
from classes import MyModel
from torch import optim, nn


def create_model(model_name="resnet18"):
    if model_name=="resnet18":
        return MyModel(models.resnet18())


def setup_optimizer(model_params, lr):
    return optim.Adam(model_params, lr=lr)

