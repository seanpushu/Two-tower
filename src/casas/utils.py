from typing import Optional

import torch


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim=None, keepdim=False):
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    mask = mask.float()
    masked_tensor = tensor * mask
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1e-6)
    return masked_tensor.sum(dim=dim, keepdim=keepdim) / denom


def masked_sum(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim=None, keepdim=False):
    if mask is None:
        return tensor.sum(dim=dim, keepdim=keepdim)
    mask = mask.float()
    return (tensor * mask).sum(dim=dim, keepdim=keepdim)


def apply_key_padding_mask(attn_mask: torch.Tensor, key_padding_mask: Optional[torch.Tensor]):
    if key_padding_mask is None:
        return attn_mask
    attn_mask = attn_mask.clone()
    attn_mask = attn_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    return attn_mask

