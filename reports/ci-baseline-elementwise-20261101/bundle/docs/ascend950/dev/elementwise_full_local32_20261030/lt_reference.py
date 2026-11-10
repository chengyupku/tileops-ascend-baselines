"""CPU less-than, with bool storage interpreted by nonzero truth."""
import torch
def lt_reference(a,b):
 return torch.lt(a.view(torch.uint8)!=0,b.view(torch.uint8)!=0) if a.dtype==torch.bool else torch.lt(a,b)
