"""Shared, swappable feature extractors for the CIFAR-10 retrieval pipeline.

Two extractors are provided so the retrieval pipeline can plug in either one
without any other changes:

* ``BaselineExtractor`` -- the original baseline (member 1): a pretrained
  ImageNet ResNet18 with the classifier removed, producing L2-normalized
  512-d features.
* ``EmbeddingNet`` -- the metric-learning model (member 2): the same ResNet18
  backbone followed by a projection head mapping 512 -> 128, trained with the
  batch-hard triplet loss in ``losses.py``.

``build_feature_extractor(mode, ...)`` returns ``(model, transform)`` and is the
single entry point used by ``retrieval.py`` (and reusable by member 3 for the
part c) evaluation). Both extractors return already L2-normalized embeddings,
so callers can rank by a plain dot product (cosine similarity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ImageNet statistics used to normalize inputs for the fine-tuned model.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EMBEDDING_DIM = 128


def get_device():
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resnet18_backbone(pretrained=True):
    """ResNet18 with the final classifier removed -> output shape (B, 512, 1, 1)."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    return nn.Sequential(*list(backbone.children())[:-1])


class BaselineExtractor(nn.Module):
    """Pretrained ResNet18 backbone returning L2-normalized 512-d features.

    Reproduces the baseline behavior from the original ``retrieval.py``.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = _resnet18_backbone(pretrained=pretrained)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # (B, 512)
        return F.normalize(feat, p=2, dim=1)


class EmbeddingNet(nn.Module):
    """ResNet18 backbone + projection head (512 -> ``embedding_dim``).

    The projection head is a single linear layer followed by a BatchNorm
    (BNNeck-style, which is known to help retrieval). The forward pass returns
    L2-normalized embeddings so that Euclidean distance and cosine similarity
    are equivalent for both training (triplet loss) and retrieval.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = _resnet18_backbone(pretrained=pretrained)
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # (B, 512)
        emb = self.projection(feat)          # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)


def baseline_transform():
    """Preprocessing for the baseline extractor (matches the original code)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def trained_transform():
    """Preprocessing for the fine-tuned model (adds ImageNet normalization).

    64x64 is enough for CIFAR-10 metric learning and ~10x faster to train than
    224x224 (CIFAR-10 images are natively 32x32, so 224 adds no information).
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_feature_extractor(mode="trained", ckpt_path=None, device=None):
    """Build a feature extractor and its matching eval-time transform.

    Args:
        mode: ``"baseline"`` for the pretrained ResNet18 (512-d) or
            ``"trained"`` for the triplet-trained ``EmbeddingNet`` (128-d).
        ckpt_path: checkpoint saved by ``train.py`` (required for ``"trained"``).
        device: target device; defaults to :func:`get_device`.

    Returns:
        ``(model, transform)`` -- model is in ``eval()`` mode on ``device``.
    """
    if device is None:
        device = get_device()

    if mode == "baseline":
        model = BaselineExtractor(pretrained=True)
        transform = baseline_transform()
    elif mode == "trained":
        if ckpt_path is None:
            raise ValueError("ckpt_path is required when mode='trained'")
        ckpt = torch.load(ckpt_path, map_location=device)
        embedding_dim = ckpt.get("embedding_dim", EMBEDDING_DIM)
        # Backbone weights are overwritten by the checkpoint, so skip the
        # (slow) pretrained download here.
        model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        transform = trained_transform()
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'baseline' or 'trained')")

    model = model.to(device)
    model.eval()
    return model, transform
