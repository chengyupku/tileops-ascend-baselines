import sys,runpy,dataclasses,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-old1076-20261029/arm_graph_root'
name='benchmarks.ascend_all_pairs.ops_transformer';package=types.ModuleType(name);package.__path__=[str(ROOT/'reports/baseline-old1076-20261029/arm_fp8_adapters')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_transformer_graph import _candidate as original_candidate
def _candidate(binding,run,op,directory,adaptation,host=False):
 c=original_candidate(binding,run,op,directory,adaptation)
 return dataclasses.replace(c,timing_mode='host_roundtrip' if host else c.timing_mode)
from benchmarks.ascend_all_pairs.ops_transformer import native
native.WORKSPACE=ROOT/"reports/baseline-old1076-20261029/arm_fp8_root"
def quant_gemm(case):
    import torch
    from benchmarks.ascend_all_pairs.ops_transformer.native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    op=case._call.signature.name;p=case.params;batched=op=='BmmFP8FwdOp'
    a,b,sa,sb,*extra=case.inputs
    m=a.shape[-2];n=b.shape[-1] if batched and not p['trans_b'] else b.shape[-2]
    if sa.numel() not in (1,m) or sb.numel() not in (1,n):
        raise UnsupportedCase('This original GroupedMatmul route selects T-T/K-C scale shapes; blockwise scale packing is separately unresolved, never replaced by torch dequantization.')
    if extra and extra[0] is not None and op!='GemmINT8W8A8FwdOp':
        raise UnsupportedCase('Original FP8 GroupedMatmul dynamic quantization mode requires null bias.')
    binding=GraphCall('ops-transformer')
    def run(a,b,sa,sb,bias=None):
        outputs=[]
        pairs=zip(a,b,strict=True) if batched else [(a,b)]
        for lhs,rhs in pairs:
            w=rhs if batched and not p['trans_b'] else rhs.T
            scale_a=sa.reshape(-1).expand(lhs.shape[0]).contiguous() if sa.numel()==1 else sa.flatten().contiguous()
            scale_b=sb.reshape(-1).expand(w.shape[1]).contiguous().reshape(1,-1) if sb.numel()==1 else sb.contiguous().reshape(1,-1)
            groups=torch.full((1,),lhs.shape[0],dtype=torch.int64,device=a.device)
            out=torch.empty((lhs.shape[0],w.shape[1]),dtype=torch.float16 if p['out_dtype']=='bfloat16' else getattr(torch,p['out_dtype']),device=a.device)
            binding.call('GroupedMatmul',[lhs.contiguous(),w.contiguous()[None],None if bias is None else bias.contiguous().reshape(1,-1),scale_b,None,None,None,groups,scale_a],[out],
                {'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},
                names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
            outputs.append(out.to(getattr(torch,p['out_dtype'])))
        return torch.stack(outputs) if batched else outputs[0]
    return _candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Original Ascend950 FP8/INT8 dynamic quantization, one group, exact tensor/token/channel scale expansion and weight transpose; complete native scaled dot product and optional original bias.')

def _fp8_block_product(binding,a,b,sa,sb,dtype):
    import torch
    m=a.shape[0];n=b.shape[0];out=torch.empty((m,n),dtype=torch.float16 if dtype==torch.bfloat16 else dtype,device=a.device)
    if not m:return out
    groups=torch.full((1,),m,dtype=torch.int64,device=a.device)
    binding.call('GroupedMatmul',[a.contiguous(),b.T.contiguous()[None],None,sb.T.contiguous()[None],None,None,None,groups,sa.contiguous()],[out],{'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
    return out

def fp8_block_gemm(case):
    import torch
    from benchmarks.ascend_all_pairs.ops_transformer.native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    moe=case._call.signature.name=='MoEGroupedGemmFP8FwdOp';binding=GraphCall('ops-transformer')
    if not moe:
        a,b,sa,sb=case.inputs;m,k=a.shape;n=b.shape[0]
        if sa.shape!=(m,-(-k//128)) or sb.shape!=(-(-n//128),-(-k//128)):
            raise UnsupportedCase('Original G-B supports activation per row/K128 and weight N128/K128; independent per-weight-row/K128 scales are not this native mode.')
        dtype=getattr(torch,case.params['out_dtype'])
        def run(a,b,sa,sb):return _fp8_block_product(binding,a,b,sa,sb,dtype)
    else:
        if next(iter(case.params['layout']))!='masked':raise UnsupportedCase('Original FP8 route currently binds manifest masked expert layout.')
        dtype=getattr(torch,case._call.specs['output'].dtype)
        def run(a,sa,b,sb,metadata):
            out=torch.zeros((*a.shape[:2],b.shape[1]),dtype=dtype,device=a.device)
            for i,count in enumerate(metadata.tolist()):
                if count:out[i,:count]=_fp8_block_product(binding,a[i,:count],b[i],sa[i,:count],sb[i],dtype)
            return out
    return _candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Original Ascend950 G-B FP8 mode, exact M/K128 and N128/K128 scale layouts, scale transpose and singleton native group; no dequantization outside the original kernel. Masked experts use native products for live rows and preserve zero output padding.',host=moe)

def fp8_gemm_dispatch(case):
    a,b,sa,sb,*_=case.inputs
    if sa.numel() not in (1,a.shape[-2]) or sb.numel() not in (1,b.shape[-2]):return fp8_block_gemm(case)
    return quant_gemm(case)

from benchmarks.ascend_baselines import source_hashes
def factory(case,library):
 if library!='ops-transformer':raise ValueError('Transformer only')
 def build():
  name=case._call.signature.name
  fn=fp8_block_gemm if name=='MoEGroupedGemmFP8FwdOp' else quant_gemm if name in ('BmmFP8FwdOp','GemmINT8W8A8FwdOp') else fp8_gemm_dispatch
  c=fn(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__),*(ROOT/'reports/baseline-old1076-20261029/arm_fp8_adapters').glob('*.py'))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
