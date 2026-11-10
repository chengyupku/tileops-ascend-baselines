"""Exact-scope original SGL complete API bindings, no kernel changes."""
from pathlib import Path
import hashlib
from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
ROOT=Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source')
HOST='python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py'

def build(case):
    import torch
    from benchmarks.api import Implementation
    p=case.params
    is_gdn=len(case.inputs)==10
    if any(p.get(k,False) for k in ('use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
        raise UnsupportedCase('Native/chunk complete APIs do not implement fused gate/beta flags; old chunk_kda route excluded.')
    l2=p.get('use_qk_l2norm_in_kernel',False)
    name='chunk_gated_delta_rule_npu' if l2 else 'chunk_gated_delta_rule_native'
    import importlib, inspect, sys
    sys.path.insert(0,str(ROOT/'python/sgl_kernel_npu'))
    module=importlib.import_module('sgl_kernel_npu.fla.chunk')
    fn=getattr(module,name)
    assert Path(module.__file__).resolve()==(ROOT/HOST).resolve()
    assert Path(inspect.unwrap(fn).__code__.co_filename).resolve()==(ROOT/HOST).resolve()
    q=case.inputs[0]
    if p.get('scale') not in (None,q.shape[-1]**-0.5):
        raise UnsupportedCase('Native fixed scale differs from contract.')
    def run(*inputs):
        if is_gdn:
            q,k,v,g,beta,state,offsets,offsets_cpu,A_log,dt_bias=inputs
        else:
            q,k,v,beta,state,offsets,offsets_cpu=inputs
            g=torch.zeros_like(beta)
        if q.shape[2]!=v.shape[2]:
            q=q.repeat_interleave(v.shape[2]//q.shape[2],dim=2)
            k=k.repeat_interleave(v.shape[2]//k.shape[2],dim=2)
        if state is not None and p.get('state_v_first',False): state=state.transpose(-1,-2).contiguous()
        if l2:
            out,final,_=fn(q,k,v,g.to(torch.float32),beta,initial_state=state,output_final_state=True,cu_seqlens=offsets,use_qk_l2norm_in_kernel=True)
        elif offsets is None:
            out,final=fn(q,k,v,g,beta,initial_state=state,output_final_state=True)
        else:
            bounds=(offsets_cpu if offsets_cpu is not None else offsets.cpu()).tolist()
            outputs=[]; states=[]
            for i,(a,b) in enumerate(zip(bounds,bounds[1:])):
                o,s=fn(q[:,a:b],k[:,a:b],v[:,a:b],g[:,a:b],beta[:,a:b],initial_state=None if state is None else state[i:i+1],output_final_state=True)
                outputs.append(o); states.append(s)
            out=torch.cat(outputs,dim=1); final=torch.cat(states,dim=0)
        if p.get('state_v_first',False): final=final.transpose(-1,-2).contiguous()
        return out,final
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.chunk.'+name,'Original complete API; exact zero Delta gate/nonzero GDN gate, head replication, independent sequence binding and state layout only.',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sorted((ROOT/'python/sgl_kernel_npu/sgl_kernel_npu/fla').glob('*.py'))]})
PROVIDERS={'GDNFwdOp':build,'DeltaNetFwdOp':build}
