"""CPU torch.pow reference preserving the original workload contract."""
import torch

def pow_reference(a, b):
    assert a.device.type == b.device.type == "cpu"
    return torch.pow(a, b)
