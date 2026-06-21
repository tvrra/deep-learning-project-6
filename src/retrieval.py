import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from model import build_feature_extractor, get_device

# Swap the feature extractor here -- this is the only line that changes between
# the baseline (member 1) and the trained metric-learning model (member 2).
#   "baseline" -> pretrained ResNet18, 512-d L2-normalized features
#   "trained"  -> triplet-trained EmbeddingNet, 128-d L2-normalized embeddings
MODE = "trained"
CKPT_PATH = "checkpoints/triplet_resnet18.pt"

device = get_device()
print("Using device:", device)

# build_feature_extractor returns the model and its matching preprocessing.
# Both extractors already L2-normalize their output, so retrieval just ranks by
# cosine similarity (a dot product) below.
model, transform = build_feature_extractor(
    MODE,
    ckpt_path=CKPT_PATH if MODE == "trained" else None,
    device=device,
)

gallery_dataset = CIFAR10(root="data", train=True, download=False, transform=transform)
query_dataset = CIFAR10(root="data", train=False, download=False, transform=transform)

gallery_subset = Subset(gallery_dataset, range(5000))
query_subset = Subset(query_dataset, range(100))

gallery_loader = DataLoader(gallery_subset, batch_size=32, shuffle=False)
query_loader = DataLoader(query_subset, batch_size=32, shuffle=False)


def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(loader):
            images = images.to(device)
            outputs = model(images)  # already flattened + L2-normalized

            features.append(outputs.cpu())
            labels.append(batch_labels)

    return torch.cat(features), torch.cat(labels)


gallery_features, gallery_labels = extract_features(gallery_loader)
query_features, query_labels = extract_features(query_loader)

similarities = query_features @ gallery_features.T

top_k = 5
topk_values, topk_indices = torch.topk(similarities, k=top_k, dim=1)

for i in range(len(query_subset)):
    print("\nQuery", i)
    print("Query label:", query_labels[i].item())

    retrieved_labels = gallery_labels[topk_indices[i]]
    print("Retrieved labels:", retrieved_labels.tolist())


#save results into files, so we can access them easily

torch.save(gallery_features, f"gallery_features_{MODE}.pt")
torch.save(gallery_labels, f"gallery_labels_{MODE}.pt")
torch.save(query_features, f"query_features_{MODE}.pt")
torch.save(query_labels, f"query_labels_{MODE}.pt")
