import torch
import yaml

from src.models.vit_model import ViTClassifier
from src.models.fusion_model import FusionModel

def main():
    with open("experiments/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["use_tda"]:
        model = FusionModel()
    else:
        model = ViTClassifier()

    model.to(device)

    print("Model initialized:", model.__class__.__name__)

if __name__ == "__main__":
    print('Hello world')
    main()
