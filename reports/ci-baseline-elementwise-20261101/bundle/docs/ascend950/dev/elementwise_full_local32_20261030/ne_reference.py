"""CPU equality, with bool storage interpreted by nonzero truth."""
import torch
def ne_reference(a,b):
 return torch.ne(a.view(torch.uint8)!=0,b.view(torch.uint8)!=0) if a.dtype==torch.bool else torch.ne(a,b)
