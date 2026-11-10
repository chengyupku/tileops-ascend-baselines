"""Independent complete original LayerNorm calls bind per-channel InstanceNorm."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def instance(case):
    import torch,math
    from integration700_transformer_dense_graph import GraphCall
    from benchmarks.ascend_baselines import Candidate,source_hashes,UnsupportedCase
    from benchmarks.api import Implementation
    x,rm,rv,w,bias=case.inputs
    if not case.params['use_input_stats'] or rm is not None:raise UnsupportedCase('Original NormRopeConcat has no running-statistics state contract')
    width=math.prod(x.shape[2:])
    if width>1024 or width%2:raise UnsupportedCase('Complete reduction width outside original API limit')
    vendor=P/'rope/classicpilot_attention_transformer'
    b=GraphCall('ops-transformer')
    def run(x,running_mean,running_var,weight,bias):
        results=[]
        for i in range(x.shape[1]):
            inp=x[:,i].float().contiguous().reshape(1,x.shape[0],1,width)
            gamma=weight[i].float().expand(width).contiguous();beta=bias[i].float().expand(width).contiguous()
            enc=torch.empty((1,0,1,width),dtype=torch.float32,device=x.device);rt=torch.empty((0,width),dtype=torch.float32,device=x.device)
            one=torch.ones_like(gamma);zero=torch.zeros_like(gamma);outs=[torch.empty((1,1,x.shape[0],width),dtype=torch.float32,device=x.device) for _ in range(3)]
            ins=[inp,inp,inp,enc,enc,enc,gamma,beta,gamma,beta,one,zero,one,zero,rt,rt]
            b.call('NormRopeConcat',ins,outs,{'norm_type':2,'norm_added_type':0,'rope_type':0,'concat_order':0,'eps':float(case.params['eps']),'is_training':False},names=['query', 'key', 'value', 'encoder_query', 'encoder_key', 'encoder_value', 'norm_query_weight', 'norm_query_bias', 'norm_key_weight', 'norm_key_bias', 'norm_added_query_weight', 'norm_added_query_bias', 'norm_added_key_weight', 'norm_added_key_bias', 'rope_sin', 'rope_cos', 'query_output', 'key_output', 'value_output']);results.append(outs[0].reshape(x.shape[0],*x.shape[2:]))
        return torch.stack(results,1).to(x.dtype)
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h')]
    return Candidate(Implementation(run),'ops-transformer::aclnnNormRopeConcat','Independent complete channel reductions; lossless FP32 widening and affine parameter broadcasting, full output reshape/cast. No reduction splitting or extra math.',source_hashes(*files),cleanup=b.cleanup)
