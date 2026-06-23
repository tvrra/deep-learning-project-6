import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

MODE = "trained" #use "baseline" or "trained"



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


#5 for baseline:
five_best_queries_base = [30, 53, 72, 82, 94]
five_worst_queries_base = [35, 70, 87, 58, 3]

#5 for trained:
five_best_queries_train = [3, 12, 25, 48, 94]
five_worst_queries_train = [37, 47, 58, 33, 57]

#10 worst for base
ten_worst_queries_base = [37, 47, 57, 58]

#10 wrost for trained
ten_worst_queries_train = [0, 24, 59, 77]


#querries where results very different
same_queries = [42, 47, 58]

#interesting cases
interesting = [37, 58]



#### matrix
num_classes = 10
cm = torch.zeros((num_classes, num_classes))
k = 5

# alle Queries durchgehen
for i in range(len(query_labels)):
    q_label = query_labels[i]
    retrieved = topk_indices[i, :k]

    for g_idx in retrieved:
        g_label = gallery_labels[g_idx]
        cm[q_label, g_label] += 1

cm = cm / (cm.sum(dim=1, keepdim=True) + 1e-8)

"""
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
"""

def plot_retrieval_examples(query_ids, title, k=5):

    fig = plt.figure(figsize=(2 * (k + 1), 2 * len(query_ids)))

    for row, idx in enumerate(query_ids):

        # Query-Bild
        ax = fig.add_subplot(len(query_ids), k + 1, row * (k + 1) + 1,
                             xticks=[], yticks=[])

        query_img = query_dataset[idx][0]
        ax.imshow(query_img.numpy().transpose(1, 2, 0))

        ax.set_title(
            query_dataset.classes[query_labels[idx]],
            fontsize=10
        )

        # Top-k Retrievals
        for col in range(k):

            gallery_idx = topk_indices[idx][col]
            hit_img = gallery_dataset[gallery_idx][0]

            ax = fig.add_subplot(
                len(query_ids),
                k + 1,
                row * (k + 1) + col + 2,
                xticks=[],
                yticks=[]
            )

            ax.imshow(hit_img.numpy().transpose(1, 2, 0))

            retrieved_label = gallery_labels[gallery_idx]
            correct = retrieved_label == query_labels[idx]

            ax.set_title(
                gallery_dataset.classes[retrieved_label],
                color="green" if correct else "red",
                fontsize=9
            )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

plot_retrieval_examples(interesting, 'Visualization of interesting cases' +  ' (' + MODE + ')', k=10)




plt.figure(figsize=(6, 5))
plt.imshow(cm.numpy(), cmap="viridis")
plt.colorbar()
plt.title("Retrieval Confusion Matrix (Top-5)")
plt.xlabel("Retrieved label")
plt.ylabel("Query label")
plt.show()

print(gallery_labels)
