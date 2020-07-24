import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes=10, pretrained=True):
    # [REF] https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#resnet
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)

    return model
