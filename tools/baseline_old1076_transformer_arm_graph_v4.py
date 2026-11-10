import sys,runpy,dataclasses,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-old1076-20261029/arm_graph_root'
name='benchmarks.ascend_all_pairs.ops_transformer';package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_transformer_graph import _candidate as original_candidate
def _candidate(binding,run,op,directory,adaptation,host=False):
 c=original_candidate(binding,run,op,directory,adaptation)
 return dataclasses.replace(c,timing_mode='host_roundtrip' if host else c.timing_mode)
def _gemm(binding,a,b,out_dtype):
    import torch
    if not a.shape[0] or not a.shape[1]:
        return torch.zeros((a.shape[0],b.shape[1]),dtype=out_dtype,device=a.device)
    out=torch.zeros((1,a.shape[0],b.shape[1]),dtype=torch.float32,device=a.device)
    group=torch.full((1,),a.shape[1],dtype=torch.int64,device=a.device)
    binding.call('GroupedMatmulAdd',[a.T.contiguous(),b.contiguous(),group,out],[out],{'transpose_x':True,'transpose_weight':False,'group_type':2,'group_list_type':0})
    binding.cleanup()  # Original synchronized cleanup after each completed independent group.
    return out[0].to(out_dtype)

def grouped_gemm(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;moe=case._call.signature.name=='MoEGroupedGemmFwdOp'
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('GroupedMatmulAdd selects original FP16/BF16 inputs.')
    if moe and p.get('activation') is not None:raise UnsupportedCase('Native GroupedMatmulAdd has no gated activation epilogue.')
    binding=GraphCall('ops-transformer')
    def run(a,b,metadata,offsets=None):
        sizes=metadata.tolist();out_dtype=getattr(torch,p['out_dtype']) if p.get('out_dtype') else a.dtype
        if not moe:
            outputs=[];start=0
            for i,n in enumerate(sizes):
                end=start+n
                if p['transpose_a']:
                    lhs=a[start:end].T;rhs=b[:,start:end].T if p['transpose_b'] else b[start:end]
                else:
                    lhs=a[start:end];rhs=b[i].T if p['transpose_b'] else b[i]
                outputs.append(_gemm(binding,lhs,rhs,out_dtype));start=end
            return torch.stack(outputs) if p['transpose_a'] else torch.cat(outputs)
        layout=p['layout'];kind=next(iter(layout));layout=layout[kind]
        out=torch.zeros((*a.shape[:-1],b.shape[1]),dtype=out_dtype,device=a.device)
        if kind=='masked':
            for i,n in enumerate(sizes):
                if n:out[i,:n]=_gemm(binding,a[i,:n],b[i].T,out_dtype)
        elif layout['metadata_kind']=='per_row':
            for i in range(b.shape[0]):
                indices=(metadata==i).nonzero().flatten()
                if indices.numel():out[indices]=_gemm(binding,a[indices],b[i].T,out_dtype)
        else:
            start=0
            for i,end in enumerate(sizes):
                align=layout['alignment'];start=-(-start//align)*align
                if end>start:out[start:end]=_gemm(binding,a[start:end],b[i].T,out_dtype)
                start=end
        return out
    return _candidate(binding,run,'GroupedMatmulAdd','gmm/grouped_matmul_add','Per-call original group metadata selects independent slices; call original GEMM for each group, assemble outputs and preserve zero padding.',host=True)
from benchmarks.ascend_baselines import source_hashes
def factory(case,library):
 if library!='ops-transformer' or case._call.signature.name not in ('GroupedGemmFwdOp','MoEGroupedGemmFwdOp'):raise ValueError('BMM pilot only')
 def build():
  c=grouped_gemm(case);return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
