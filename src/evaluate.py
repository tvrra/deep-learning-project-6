from metrics import recall_at_k, mean_average_precision
import torch


gallery_features = torch.load("gallery_features.pt")
gallery_labels = torch.load("gallery_labels.pt")
query_features = torch.load("query_features.pt")
query_labels = torch.load("query_labels.pt")





similarities = query_features @ gallery_features.T
topk_values, topk_indices = torch.topk(similarities, k=10, dim=1)


#print("gallery_features shape:", gallery_features.shape)
#print("query_features shape:", query_features.shape)
#print("gallery_labels shape:", gallery_labels.shape)
#print("query_labels shape:", query_labels.shape)
#print("topk_indices shape:", topk_indices.shape)



recall1 = recall_at_k(topk_indices, query_labels, gallery_labels, k=1)
recall5 = recall_at_k(topk_indices, query_labels, gallery_labels, k=5)
recall10 = recall_at_k(topk_indices, query_labels, gallery_labels, k=10)

print(f"Recall@1:  {recall1:.4f}")
print(f"Recall@5:  {recall5:.4f}")
print(f"Recall@10: {recall10:.4f}")

mean_avg_prec = mean_average_precision(topk_indices,query_labels,gallery_labels)

print(f"Mean-Average Precision (mAP):  {mean_avg_prec:.4f}")