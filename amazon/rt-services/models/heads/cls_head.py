import torch, torch.nn as nn
from ..registry import register_head

@register_head("temporal_pool_cls")
class TemporalPoolClassifier(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x, attn_mask=None):  # x: (B,T,C)
        if attn_mask is not None:
            # attn_mask: True=pad; False=valid
            lengths = (~attn_mask).sum(1).clamp(min=1).unsqueeze(-1)  # (B,1)
            x = (x.masked_fill(attn_mask.unsqueeze(-1), 0.0)).sum(1) / lengths
        else:
            x = x.mean(1)
        return self.fc(x)  # (B,num_classes)
