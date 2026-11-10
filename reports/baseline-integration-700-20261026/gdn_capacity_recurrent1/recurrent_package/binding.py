"""Parameterized original full RecurrentKda ABI, one call for the full dense case."""
import hashlib,json
from pathlib import Path
from direct_acl import DirectACL,require_guard
HERE=Path(__file__).resolve().parent

def build(case):
    require_guard()
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    eligible=json.loads((HERE.parent/'RECURRENT_CASES.json').read_text())
    contract=json.loads((HERE.parent/'CONTRACTS.json').read_text())[case.id]
    if case.id not in eligible or json.loads(json.dumps(case.params)) != contract['params']:raise ValueError('Exact authorized contract required')
    p=case.params
    runtime=json.loads((HERE/'RUNTIME_RECIPE.json').read_text())
    transport=DirectACL('RecurrentKda',torch,runtime['opapi'],runtime['nnopbase'])
    def run(q,k,v,g,beta,initial,offsets,offsets_cpu,a_log,dt_bias):
        if offsets is not None or offsets_cpu is not None:raise ValueError('dense only')
        b,s,h,dk=q.shape; hv,dv=v.shape[-2:];vf=p['state_v_first']
        if q.dtype != torch.bfloat16 or s>8:raise ValueError('native contract')
        if dk<128:
            q=torch.nn.functional.pad(q,(0,128-dk));k=torch.nn.functional.pad(k,(0,128-dk))
        if dv<128:v=torch.nn.functional.pad(v,(0,128-dv))
        state_shape=(b,hv,128,128)
        if initial is None:state=torch.zeros(state_shape,dtype=torch.float32,device=q.device)
        else:
            widths=(0,128-dk,0,128-dv) if vf else (0,128-dv,0,128-dk)
            state=torch.nn.functional.pad(initial,widths).contiguous().clone() if dk<128 or dv<128 else initial.contiguous().clone()
        gate=g[...,None].expand(-1,-1,-1,128).contiguous()
        bias=None if dt_bias is None else dt_bias[:,None].expand(-1,128).contiguous().reshape(-1)
        out,final=torch.empty_like(v),torch.empty_like(state)
        transport.call([q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.contiguous(),state,None,None,a_log,bias,None],[out,final],[b'BSND',float(dk**-.5 if p['scale'] is None else p['scale']),True,False,False,p['use_gate_in_kernel'],p['use_beta_sigmoid_in_kernel'],p['allow_neg_eigval'],False,-5.,vf])
        return out[...,:dv].contiguous(),(final[...,:dv,:dk] if vf else final[...,:dk,:dv]).contiguous()
    pins=json.loads((HERE/'RUNTIME_PINS.json').read_text())['files'];hashes={x['path']:x['sha256'] for x in pins}
    for n in ('binding.py','direct_acl.py','entry.py'):
        x=HERE/n;hashes[str(x)]=hashlib.sha256(x.read_bytes()).hexdigest()
    return Candidate(Implementation(run),'ops-transformer::aclnnRecurrentKda','One whole original ABI call; original scalar gate broadcast, optional zero padding and crop, private state, native flags and full outputs.',hashes,timing_mode='host_roundtrip',cleanup=transport.cleanup)
