import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def freeze_backbone(self):
        for name, p in self.model.named_parameters():
            p.requires_grad = name.startswith("fc")

    def unfreeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
