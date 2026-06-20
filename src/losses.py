"""Batch-hard triplet loss with in-batch hard negative mining.

Implements the "batch-hard" strategy from Hermans et al., *In Defense of the
Triplet Loss for Person Re-Identification* (https://arxiv.org/abs/1703.07737):

For every sample in the batch (used as an anchor) we mine, **within the same
batch**, the hardest positive (the same-class sample that is *farthest* away)
and the hardest negative (the different-class sample that is *closest*). The
loss pushes the hardest positive closer than the hardest negative by at least a
margin.

Because CIFAR-10 has only 10 classes, an ordinary shuffled batch of e.g. 128
images already contains many samples per class, so every anchor has valid
positives and negatives -- no dedicated P x K sampler is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """Triplet margin loss with batch-hard mining.

    Args:
        margin: desired separation between the hardest positive and hardest
            negative distance.
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """Compute the batch-hard triplet loss.

        Args:
            embeddings: (B, D) tensor, expected to be L2-normalized.
            labels: (B,) integer class labels.

        Returns:
            ``(loss, active_fraction)`` where ``active_fraction`` is the share
            of anchors whose triplet still violates the margin (a useful
            training signal: it should fall as the model improves).
        """
        # Pairwise Euclidean distances between all embeddings in the batch.
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        labels = labels.view(-1, 1)
        same = labels == labels.t()                       # (B, B) same-class mask
        eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)

        positive_mask = same & ~eye   # same class, excluding self
        negative_mask = ~same         # different class

        # Hardest positive: largest distance among same-class samples.
        # Invalid entries set to -inf so they never win the max.
        pos_dist = dist.masked_fill(~positive_mask, float("-inf"))
        hardest_pos, _ = pos_dist.max(dim=1)

        # Hardest negative: smallest distance among different-class samples.
        # Invalid entries set to +inf so they never win the min.
        neg_dist = dist.masked_fill(~negative_mask, float("inf"))
        hardest_neg, _ = neg_dist.min(dim=1)

        # Only keep anchors that actually have both a positive and a negative
        # in the batch (guards against degenerate batches).
        valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
        if valid.sum() == 0:
            return embeddings.new_zeros(()), embeddings.new_zeros(())

        hardest_pos = hardest_pos[valid]
        hardest_neg = hardest_neg[valid]

        losses = F.relu(hardest_pos - hardest_neg + self.margin)
        loss = losses.mean()
        active_fraction = (losses > 0).float().mean()
        return loss, active_fraction
