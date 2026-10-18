"""Cross entropy + clustering - separation (paper equations 8-11)."""
import torch
from torch.nn import functional as F


def prototype_loss(logits, min_distances, targets, prototype_class_identity,
                   cluster_weight=0.8, separation_weight=0.08):
    correct = prototype_class_identity[:, targets].T.bool()
    cluster = min_distances.masked_fill(~correct, torch.inf).min(dim=1).values.mean()
    # Eq. 10 defines a negative separation cost. Apply its sign exactly once.
    separation = -min_distances.masked_fill(correct, torch.inf).min(dim=1).values.mean()
    cross_entropy = F.cross_entropy(logits, targets)
    total = cross_entropy + cluster_weight * cluster + separation_weight * separation
    return total, dict(cross_entropy=cross_entropy, cluster=cluster, separation=separation)
