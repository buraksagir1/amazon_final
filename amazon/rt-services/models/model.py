import torch, torch.nn as nn
from .registry import BACKBONES, HEADS

class SignModel(nn.Module):
    """
    Genelleştirilebilir model: backbone + head.
    config örn:
    {
      "in_features": 360,  # (V*4), örn V=90 -> 360
      "backbone": {"name":"tcn", "args":{"hidden":256, "n_layers":4}},
      "head": {"name":"temporal_pool_cls", "args":{"num_classes":16}},
    }
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        bb_name = config["backbone"]["name"]
        hd_name = config["head"]["name"]
        in_feats = config["in_features"]

        self.backbone = BACKBONES[bb_name](in_ch=in_feats, **config["backbone"].get("args", {}))
        # backbone çıkış kanalını tahmin et (proj/conv son kanal)
        with torch.no_grad():
            dummy = torch.zeros(1, 8, in_feats)
            out = self.backbone(dummy)
            feat_dim = out.shape[-1]
        self.head = HEADS[hd_name](in_ch=feat_dim, **config["head"].get("args", {}))

    def forward(self, x, attn_mask=None):  # x: (B,T,V*4)
        x = self.backbone(x)
        logits = self.head(x, attn_mask=attn_mask)
        return logits

def build_model(config: dict) -> SignModel:
    return SignModel(config)

def save_checkpoint(path: str, model: nn.Module, optimizer=None, epoch=None, extra: dict | None=None):
    obj = {
        "config": getattr(model, "config", None),
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        obj["epoch"] = epoch
    if extra:
        obj["extra"] = extra
    import torch
    torch.save(obj, path)

def load_from_checkpoint(path: str, map_location="cpu") -> SignModel:
    import torch
    ckpt = torch.load(path, map_location=map_location)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
