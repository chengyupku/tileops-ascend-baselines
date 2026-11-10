def factory(case,op,native,tl,out):
 if op=='CausalConv1dFwdOp':
  from state_adapter_pto import factory as f
 else:
  from adapter_a5_bound import factory as f
 return f(case,op,native,tl,out)
