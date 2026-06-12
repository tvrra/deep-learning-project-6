import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

gallery_dataset = CIFAR10(root="data", train=True, download=False, transform=transform)
query_dataset = CIFAR10(root="data", train=False, download=False, transform=transform)

gallery_subset = Subset(gallery_dataset, range(1000))
query_subset = Subset(query_dataset, range(10))

gallery_loader = DataLoader(gallery_subset, batch_size=32, shuffle=False)
query_loader = DataLoader(query_subset, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()


def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)
            outputs = F.normalize(outputs, p=2, dim=1)

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