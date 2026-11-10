"""Independent CPU torch.lerp reference with construction-time scalar weight."""
import torch
def lerp_scalar_reference(a,b,weight=0.5):
    assert a.device.type == b.device.type == "cpu"
    return torch.lerp(a,b,weight)
