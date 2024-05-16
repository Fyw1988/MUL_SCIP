import torch 
import torch.nn as nn

from torch.nn import functional as F


def ce_loss(logits, targets, mask, reduction=True):
    log_pred = F.log_softmax(logits, dim=1)
    nll_loss = F.nll_loss(log_pred, targets, reduction='none')
    # import pdb; pdb.set_trace()
    return ((nll_loss * mask).sum() / (mask.sum() + 1e-16))


def mse_loss(logits, targets, mask, reduction=True):
    input_softmax = F.softmax(logits, dim=1)
    target_softmax = F.softmax(targets, dim=1)
    consistency_dist = (input_softmax - target_softmax) ** 2

    # return ((consistency_dist * mask).sum() / (mask.sum() + 1e-16))
    return ((consistency_dist * mask).sum() / mask.sum())
