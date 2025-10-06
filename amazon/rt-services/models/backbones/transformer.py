import torch, torch.nn as nn
from ..registry import register_backbone

@register_backbone("transformer")
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, attn_mask=None):  # x: (B,T,Feat)
        x = self.proj(x)
        return self.enc(x, src_key_padding_mask=attn_mask)  # (B,T,d_model)
