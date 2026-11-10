"""CPU equality, with bool storage interpreted by nonzero truth."""
import torch
def eq_reference(a,b):
 return torch.eq(a.view(torch.uint8)!=0,b.view(torch.uint8)!=0) if a.dtype==torch.bool else torch.eq(a,b)
