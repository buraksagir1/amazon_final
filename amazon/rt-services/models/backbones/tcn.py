import torch, torch.nn as nn
from . import __init__  # noqa
from ..registry import register_backbone

@register_backbone("tcn")
class TemporalConvNet(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 256, n_layers: int = 4, kernel: int = 5):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(ch, hidden, kernel, padding=kernel//2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (B,T,Feat)
        x = x.transpose(1, 2)          # -> (B,Feat,T)
        y = self.net(x)                # (B,H,T)
        y = y.transpose(1, 2)          # -> (B,T,H)
        return y                        # zaman boyutunda özellikler
