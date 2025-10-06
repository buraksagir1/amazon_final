from typing import Dict, Callable

BACKBONES: Dict[str, Callable] = {}
HEADS: Dict[str, Callable] = {}

def register_backbone(name):
    def deco(cls):
        BACKBONES[name] = cls
        return cls
    return deco

def register_head(name):
    def deco(cls):
        HEADS[name] = cls
        return cls
    return deco
