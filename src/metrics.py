import numpy as np


def recall_at_k(topk_indices, query_labels, gallery_labels, k):

    relevant = 0
    for x in range(len(query_labels)):
        current = query_labels[x]
        if current in gallery_labels[topk_indices[x][:k]]: 
            relevant += 1

    return relevant/len(query_labels)



def mean_average_precision(topk_indices, query_labels, gallery_labels):
    
    avg_precisions = np.zeros_like(query_labels, dtype=float)

    for x in range(len(query_labels)):
        current = query_labels[x]
        count = 0
        temp = np.zeros_like(topk_indices[x])

        for y in range(len(topk_indices[x])):
            if current == gallery_labels[topk_indices[x][y]]:
                count += 1
                temp[y] = count/(y+1)
        

        if len(temp[temp>0]) > 0:
            avg_precisions[x] = np.average(temp[temp>0])

    return np.average(avg_precisions)
            
