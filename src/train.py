"""Fine-tune an embedding model on CIFAR-10 with the batch-hard triplet loss.

This is part b) "metric learning": we start from an ImageNet-pretrained ResNet18
and learn a 128-d embedding space (via the projection head in ``model.py``)
using the batch-hard triplet loss in ``losses.py``. The resulting checkpoint is
consumed by ``retrieval.py`` simply by swapping the feature extractor.

Run:
    python src/train.py

The script self-bootstraps the CIFAR-10 download into ``data/``.
"""

import os
import ssl

import certifi
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from losses import BatchHardTripletLoss
from model import EmbeddingNet, get_device, IMAGENET_MEAN, IMAGENET_STD

# Allow CIFAR-10 to download over HTTPS on macOS (mirrors dataset.py).
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_ROOT = "data"
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "triplet_resnet18.pt")

EMBEDDING_DIM = 128
BATCH_SIZE = 128
EPOCHS = 12
LR = 3e-4
WEIGHT_DECAY = 1e-4
MARGIN = 0.2
NUM_WORKERS = 4


def build_train_loader():
    """CIFAR-10 train split with light augmentation + ImageNet normalization.

    The transform must match ``model.trained_transform`` at inference time
    (same resize + normalization) so the embeddings are comparable.
    """
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = CIFAR10(root=DATA_ROOT, train=True, download=True,
                      transform=train_transform)
    # With only 10 classes, a shuffled batch of 128 contains every class many
    # times over, so batch-hard mining always finds valid positives/negatives.
    # (A P x K sampler would be the alternative for datasets with many classes.)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, drop_last=True)


def train():
    device = get_device()
    print("Using device:", device)

    train_loader = build_train_loader()

    model = EmbeddingNet(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)
    criterion = BatchHardTripletLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_active = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss, active = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_active += active.item()
            n_batches += 1
            progress.set_postfix(loss=loss.item(), active=active.item())

        scheduler.step()
        print(f"Epoch {epoch}: loss={running_loss / n_batches:.4f} "
              f"active_triplets={running_active / n_batches:.3f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "embedding_dim": EMBEDDING_DIM}, CKPT_PATH)
    print("Saved checkpoint to", CKPT_PATH)
    return model


if __name__ == "__main__":
    train()
