import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        return self.model(x)