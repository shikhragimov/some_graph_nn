import torch


def pooling(pos_h):
    return torch.sigmoid(pos_h.mean(dim=0, keepdim=True))
