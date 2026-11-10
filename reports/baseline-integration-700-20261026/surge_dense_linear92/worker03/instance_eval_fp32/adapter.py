"""Original BatchNormV3 evaluation API with lossless FP32 input widening."""
def make(case):
 import torch
 from benchmarks.ascend_all_pairs.ops_nn.graph import GraphCall
 from benchmarks.ascend_all_pairs.ops_nn.common import candidate
 assert case._call.signature.name=='InstanceNormFwdOp'
 p=case.params
 assert not p['use_input_stats']
 b=GraphCall('ops-nn')
 def run(x,rm,rv,w,bias):
  c=x.shape[1];xx=x.float().contiguous().reshape(x.shape[0],c,1,-1);out=torch.empty_like(xx);mean=rm.float().clone();var=rv.float().clone();stat=torch.empty_like(mean)
  b.call('BatchNormV3',[xx,w.float(),bias.float(),mean,var],[out,mean,var,stat,torch.empty_like(stat)],{'epsilon':float(p['eps']),'momentum':float(p['momentum']),'is_training':False},formats=[0,2,2,2,2,0,2,2,2,2])
  return out.reshape(x.shape).to(x.dtype),mean,var
 return candidate(b,run,'BatchNormV3','Original complete evaluation API; lossless FP16 to FP32 widening and output dtype restoration; cloned running stats.','norm/batch_norm_v3')
