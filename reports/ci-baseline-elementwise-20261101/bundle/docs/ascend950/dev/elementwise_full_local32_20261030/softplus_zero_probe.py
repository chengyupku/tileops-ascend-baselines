import torch
x=torch.tensor([0.,-0.,1.,-1.,float('inf'),-float('inf'),float('nan')])
for b in [0.,-0.]:
 for t in [20.,-1.,float('nan')]:
  y=torch.nn.functional.softplus(x,beta=b,threshold=t)
  print(b,t,y.tolist(),y.view(torch.int32).tolist(),flush=True)
