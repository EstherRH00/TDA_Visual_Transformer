import torch
import torch.nn as nn
import timm

class FusionModel(nn.Module):
    def __init__(self, tda_input_dim=1400):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.tda_fc = nn.Linear(tda_input_dim, 128)
        
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


class DualViTFusionModel(nn.Module):
    """Two ViT branches: one for the mammogram, one for the persistence image (as a 2D image).
    CLS tokens from both are concatenated and passed to a classifier."""
    def __init__(self):
        super().__init__()

        self.vit_image = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit_image.head = nn.Identity()

        self.vit_tda = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit_tda.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, tda):
        import torch.nn.functional as F
        # tda shape: (batch, C, H, W) — make 3-channel and resize to 224x224
        if tda.dim() == 3:
            tda = tda.unsqueeze(1)
        b, c, h, w = tda.shape
        if c == 2:
            zeros = torch.zeros(b, 1, h, w, device=tda.device)
            tda = torch.cat([tda, zeros], dim=1)
        elif c == 1:
            tda = tda.repeat(1, 3, 1, 1)
        tda = F.interpolate(tda, size=(224, 224), mode='bilinear', align_corners=False)

        img_features = self.vit_image(x)
        tda_features = self.vit_tda(tda)

        combined = torch.cat([img_features, tda_features], dim=1)
        return self.classifier(combined)
