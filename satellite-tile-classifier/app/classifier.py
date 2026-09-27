"""
Tile classifier loads a fine-tuned ResNet-18 and runs inference on CPU.
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from app.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, MODEL_PATH

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None


def _build_model() -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(model_path: Path = MODEL_PATH) -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    model = _build_model()
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    _model = model
    return _model


def classify_tile(image_path: Path) -> dict:
    """
    Classify a single tile image.

    Returns:
        {
            "predicted_label": str,
            "confidence": float,
            "probabilities": {class_name: float, ...}
        }
    """
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()  # (NUM_CLASSES,)

    prob_dict = {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)}
    top_idx = int(probs.argmax())

    return {
        "predicted_label": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": prob_dict,
    }