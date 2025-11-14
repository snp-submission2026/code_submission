import torch
import torch.nn.functional as F

def focal_loss(logits, targets, weight=None, gamma=1.5):
    ce = F.cross_entropy(logits, targets, reduction='none', weight=weight)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
