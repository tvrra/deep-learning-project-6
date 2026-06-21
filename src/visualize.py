import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

MODE = "trained" #or trained



gallery_features = torch.load(f"gallery_features_{MODE}.pt")
gallery_labels = torch.load(f"gallery_labels_{MODE}.pt")
query_features = torch.load(f"query_features_{MODE}.pt")
query_labels = torch.load(f"query_labels_{MODE}.pt")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

gallery_dataset = CIFAR10(root="data", train=True, download=False, transform=transform)
query_dataset = CIFAR10(root="data", train=False, download=False, transform=transform)

similarities = query_features @ gallery_features.T
topk_indices = torch.topk(similarities, k=10, dim=1).indices

plot = plt.figure(figsize=(12, 8))

for idx in range(5):
    ax = plot.add_subplot(5, 6, (idx*6)+1, xticks=[], yticks=[])
    img = query_dataset[idx][0]
    ax.set_title(query_dataset.classes[query_labels[idx]], color='black')
    ax.imshow(img.numpy().transpose(1, 2, 0))

    for q in range(5):
        hit = gallery_dataset[topk_indices[idx][q]][0]
        ax = plot.add_subplot(5, 6, idx*6 + q + 2, xticks=[], yticks=[])
        ax.imshow(hit.numpy().transpose(1, 2, 0))
        
        if query_labels[idx] == gallery_labels[topk_indices[idx][q]]:
            ax.set_title(gallery_dataset.classes[gallery_labels[topk_indices[idx][q]]], color='green')
        else:
            ax.set_title(gallery_dataset.classes[gallery_labels[topk_indices[idx][q]]], color='red')


plt.tight_layout()
plt.suptitle('Visualization of sample retrieval hits' + ' (' + MODE + ')', size=14)
plt.subplots_adjust(top=0.9)
plt.show()
