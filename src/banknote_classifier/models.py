import torch.nn as nn
import torchvision.models as models

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

@register_model("vgg16")
class BanknoteClassifierVGG(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        self.vgg = models.vgg16(weights=None)
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

@register_model("resnet18")
class BanknoteClassifierResNet(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__(num_classes)
        self.resnet = models.resnet18(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found.")
    return MODEL_REGISTRY[model_name](num_classes=num_classes)
