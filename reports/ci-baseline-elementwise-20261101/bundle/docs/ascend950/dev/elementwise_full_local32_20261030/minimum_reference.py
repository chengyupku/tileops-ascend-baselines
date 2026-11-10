"""CPU oracle for original NaN-propagating, IEEE zero-tie Minimum contract."""
import torch

def minimum_reference(a, b):
    assert a.device.type == b.device.type == 'cpu'
    if a.dtype == torch.bool:
        # A bool tensor may contain any nonzero storage byte.
        return (a.view(torch.uint8) != 0) & (b.view(torch.uint8) != 0)
    result = torch.minimum(a, b)
    if a.dtype.is_floating_point:
        # CPU torch tie selection is not the CUDA nan_max signed-zero contract.
        both_zero = (a == 0) & (b == 0)
        negative_zero = a.signbit() | b.signbit()
        zeros = torch.where(negative_zero, torch.full_like(result, -0.), torch.zeros_like(result))
        result = torch.where(both_zero, zeros, result)
        result = torch.where(result.isnan(), torch.full_like(result, float('nan')), result)
    return result
