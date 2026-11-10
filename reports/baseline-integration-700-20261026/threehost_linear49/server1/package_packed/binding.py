"""Original whole ChunkGatedDeltaRule mapping with standard ACLNN transport."""
import hashlib,json
from pathlib import Path
from direct_acl import DirectACL,require_guard
HERE=Path(__file__).resolve().parent

def build(case):
    require_guard()
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    c=json.loads((HERE.parent/'CONTRACTS.json').read_text())[case.id]
    if case.id not in json.loads((HERE.parent/'CHUNK_CASES.json').read_text()) or json.loads(json.dumps(case.params))!=c['params']:raise ValueError('Exact contract required')
    p=case.params;r=json.loads((HERE/'RUNTIME_RECIPE.json').read_text())
    transport=DirectACL('ChunkGatedDeltaRule',torch,r['opapi'],r['nnopbase'])
    def run(q,k,v,g,beta,initial,offsets,offsets_cpu,a_log,dt_bias):
        if a_log is not None or dt_bias is not None:raise ValueError('Preactivated gate required')
        b,s,h,dk=q.shape;hv,dv=v.shape[-2:]
        if offsets is None:
            lengths=torch.full((b,),s,dtype=torch.int32,device=q.device)
            sequences=b
        else:
            # Argument conversion only: native ABI consumes individual lengths.
            bounds=(offsets if offsets_cpu is None else offsets_cpu).tolist()
            assert bounds[0]==0 and bounds[-1]==b*s and all(y>x for x,y in zip(bounds,bounds[1:]))
            lengths=torch.tensor([y-x for x,y in zip(bounds,bounds[1:])],dtype=torch.int32,device=q.device)
            sequences=len(bounds)-1
        state=torch.zeros((sequences,hv,dv,dk),dtype=torch.float32,device=q.device) if initial is None else (initial if p.get('state_v_first',False) else initial.transpose(-2,-1)).contiguous().clone()
        out=torch.empty((b*s,hv,dv),dtype=v.dtype,device=v.device);final=torch.empty_like(state)
        transport.call([q.reshape(-1,h,dk).contiguous(),k.reshape(-1,h,dk).contiguous(),v.reshape(-1,hv,dv).contiguous(),beta.reshape(-1,hv).contiguous(),state,lengths,None if g is None else g.reshape(-1,hv).float().contiguous()],[out,final],[float(dk**-.5 if p['scale'] is None else p['scale'])])
        return out.reshape(b,s,hv,dv),(final if p.get('state_v_first',False) else final.transpose(-2,-1).contiguous())
    hashes={x['path']:x['sha256'] for x in json.loads((HERE/'RUNTIME_PINS.json').read_text())['files']}
    for n in ('binding.py','direct_acl.py','entry.py'):
        f=HERE/n;hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    
    if c['op']=='DeltaNetFwdOp':
        gdn_run=run
        def run(q,k,v,beta,initial,offsets,offsets_cpu):
            return gdn_run(q,k,v,None,beta,initial,offsets,offsets_cpu,None,None)
    return Candidate(Implementation(run),'ops-transformer::aclnnChunkGatedDeltaRule','Original whole token-major chunk API; exact state transpose, scalar gate widening, both outputs.',hashes,timing_mode='host_roundtrip',cleanup=transport.cleanup)
