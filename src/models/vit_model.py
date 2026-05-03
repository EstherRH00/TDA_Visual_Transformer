import torch.nn as nn
import timm

class ViTClassifier(nn.Module):
    """Binary classifier using a pretrained ViT-B/16 backbone.

    Replaces the ImageNet classification head with a single linear layer
    that outputs one logit for binary classification via BCEWithLogitsLoss.
    """
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        """Forward pass through the full ViT.

        Args:
            - x: input image tensor of shape (batch, 3, 224, 224).

        Returns:
            - logit: raw logit tensor of shape (batch, 1).
        """
        return self.model(x)