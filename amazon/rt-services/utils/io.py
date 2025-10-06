import torch, yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k,v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch
