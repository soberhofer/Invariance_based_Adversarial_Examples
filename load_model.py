import torch, torchvision

from torchvision import models
from torch import nn

def load_model(model_name, norm_layer=True):
    if model_name == 'resnet':
        return _load_resnet(norm_layer)
    elif model_name == 'mobilenet':
        return _load_mobilenet(norm_layer)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

class ImageNetNormalization(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageNetNormalization, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return torchvision.transforms.functional.normalize(x, self.mean, self.std)

def _load_resnet(norm_layer=True):
    base_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    num_ftrs_in = base_resnet.fc.in_features
    num_ftrs_out = 10
    # We only have 10 Classes in ImageNette
    #base_resnet.fc = nn.Linear(num_ftrs_in, num_ftrs_out)
    if norm_layer:
        resnet = torch.nn.Sequential(
            ImageNetNormalization(),
            base_resnet
        )
    else:
        resnet = base_resnet
    target_layers = [base_resnet.layer4[-1]]
    return resnet.eval(), target_layers

def _load_mobilenet(norm_layer=True):

    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    target_layers = [list(mobilenet.modules())[-9]]

    num_ftrs_in = mobilenet.classifier[0].in_features
    num_ftrs_out = mobilenet.classifier[0].out_features
    #mobilenet.classifier[0] = nn.Linear(num_ftrs_in, num_ftrs_out)

    num_ftrs_in = mobilenet.classifier[3].in_features
    num_ftrs_out = 10
    # We only have 10 Classes in ImageNette
    
    #mobilenet.classifier[3] = nn.Linear(num_ftrs_in, num_ftrs_out)

    if norm_layer:
        mobilenet = torch.nn.Sequential(
            ImageNetNormalization(),
            mobilenet
        )
    else:
        mobilenet = torch.nn.Sequential(
            mobilenet
        )
    
    return mobilenet.eval(), target_layers