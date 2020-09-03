import torch

from typing import Tuple

def subsample_labels(
    labels : torch.Tensor, 
    num_samples : int, 
    positive_fraction : float, 
    bg_label : int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    pos = ((labels != -1) & (labels != bg_label))
    if pos.dim()==0 : pos = pos.unsqueeze(0).nonzero().unbind(1)[0]
    else : pos = pos.nonzero().unbind(1)[0]

    neg = (labels == bg_label)
    if neg.dim()==0 : neg = neg.unsqueeze(0).nonzero().unbind(1)[0]
    else : neg = neg.nonzero().unbind(1)[0]

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(pos.numel(), num_pos)

    num_neg = num_samples - num_pos
    num_neg = min(neg.numel(), num_neg)

    perm1 = torch.randperm(pos.numel(), device=pos.device)[:num_pos]
    perm2 = torch.randperm(neg.numel(), device=neg.device)[:num_neg]

    pos_idx = pos[perm1]
    neg_idx = neg[perm2]

    return pos_idx, neg_idx