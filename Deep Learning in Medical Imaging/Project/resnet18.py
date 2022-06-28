from torch import nn
import torchvision.models as models

# -------------------------------------------------------------------------------------------------- #

class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_filters = self.model.fc.in_features
        self.layers = list(self.model.children())[:-1]
        self.backbone = nn.Sequential(*self.layers)
        self.fc1 = nn.Linear(self.num_filters, 64)
        self.fc = nn.Linear(64, 6)  

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze(3)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.fc(x)
        return x