import torch
import torch.nn as nn
from ..registry import register_backbone

def mask_to_lengths(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    attn_mask: (B,T) bool, True=pad, False=valid
    returns lengths: (B,) int
    """
    if attn_mask is None:
        raise ValueError("BiRNN backbones expect attn_mask for variable-length sequences.")
    return (~attn_mask).sum(dim=1).to(torch.int64)

class _BiRNNBase(nn.Module):
    RNN_CLS = None  # override
    def __init__(self, in_ch: int, hidden: int = 256, num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        assert self.RNN_CLS is not None
        self.proj = nn.Linear(in_ch, hidden) if in_ch != hidden else nn.Identity()
        self.rnn = self.RNN_CLS(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B,T,F), attn_mask: (B,T) bool (True=pad). Geri dönüş: (B,T,out_dim)
        """
        x = self.proj(x)  # (B,T,H)
        if attn_mask is None:
            y, _ = self.rnn(x)  # tam uzunluk
            return y
        lengths = mask_to_lengths(attn_mask)  # (B,)
        # pack → rnn → pad
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y_packed, _ = self.rnn(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True, total_length=x.size(1))  # (B,T,2H)
        return y

@register_backbone("bigru")
class BiGRUEncoder(_BiRNNBase):
    RNN_CLS = nn.GRU

@register_backbone("bilstm")
class BiLSTMEncoder(_BiRNNBase):
    RNN_CLS = nn.LSTM
