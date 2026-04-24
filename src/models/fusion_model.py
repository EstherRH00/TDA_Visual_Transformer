import torch
import torch.nn as nn
import timm

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.tda_fc = nn.Linear(100, 128)  # adjust depending on PI size

        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, tda):
        vit_features = self.vit(x)
        tda_features = self.tda_fc(tda)

        combined = torch.cat([vit_features, tda_features], dim=1)
        return self.classifier(combined)