"""CPU greater-than, with bool storage interpreted by nonzero truth."""
import torch
def gt_reference(a,b):
 return torch.gt(a.view(torch.uint8)!=0,b.view(torch.uint8)!=0) if a.dtype==torch.bool else torch.gt(a,b)
