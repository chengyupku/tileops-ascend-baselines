"""CPU greater-or-equal, with bool storage interpreted by nonzero truth."""
import torch
def ge_reference(a,b):
 return torch.ge(a.view(torch.uint8)!=0,b.view(torch.uint8)!=0) if a.dtype==torch.bool else torch.ge(a,b)
