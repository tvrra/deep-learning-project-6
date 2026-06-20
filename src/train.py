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


# ============================================================================
# --- TEMP: self-check, remove/comment before handoff to member 3 (part c) ----
# Quick Recall@K / mAP on a small subset just to confirm the trained model
# beats the baseline. Member 3 owns the real metrics in metrics.py/evaluate.py,
# so this block is intentionally self-contained and disposable.
# ============================================================================
def _self_check():
    import torch.nn.functional as F
    from torch.utils.data import Subset
    from model import build_feature_extractor

    device = get_device()
    GALLERY_N, QUERY_N, KS = 5000, 1000, (1, 5, 10)

    def extract(model, transform):
        gallery = Subset(CIFAR10(DATA_ROOT, train=True, download=True,
                                 transform=transform), range(GALLERY_N))
        query = Subset(CIFAR10(DATA_ROOT, train=False, download=True,
                               transform=transform), range(QUERY_N))
        g_loader = DataLoader(gallery, batch_size=64, num_workers=NUM_WORKERS)
        q_loader = DataLoader(query, batch_size=64, num_workers=NUM_WORKERS)

        def run(loader):
            feats, labs = [], []
            with torch.no_grad():
                for imgs, lab in tqdm(loader, desc="extract"):
                    feats.append(model(imgs.to(device)).cpu())
                    labs.append(lab)
            return torch.cat(feats), torch.cat(labs)

        return run(g_loader), run(q_loader)

    def metrics(qf, ql, gf, gl):
        sims = qf @ gf.t()                       # cosine sim (features normalized)
        ranking = sims.argsort(dim=1, descending=True)
        rel = (gl[ranking] == ql.view(-1, 1)).float()  # (Q, G) relevance
        recall = {k: (rel[:, :k].sum(dim=1) > 0).float().mean().item() for k in KS}
        # mean average precision over the full ranking
        cum_hits = rel.cumsum(dim=1)
        ranks = torch.arange(1, rel.size(1) + 1).float()
        precision_at_hits = (cum_hits / ranks) * rel
        ap = precision_at_hits.sum(dim=1) / rel.sum(dim=1).clamp(min=1)
        return recall, ap.mean().item()

    for mode in ("baseline", "trained"):
        model, transform = build_feature_extractor(
            mode, ckpt_path=CKPT_PATH if mode == "trained" else None,
            device=device)
        (gf, gl), (qf, ql) = extract(model, transform)
        recall, mAP = metrics(qf, ql, gf, gl)
        recall_str = ", ".join(f"R@{k}={recall[k]:.3f}" for k in KS)
        print(f"[{mode}] {recall_str}, mAP={mAP:.3f}")
# --- END TEMP -----------------------------------------------------------------


if __name__ == "__main__":
    train()
    _self_check()  # TEMP: remove before handoff
