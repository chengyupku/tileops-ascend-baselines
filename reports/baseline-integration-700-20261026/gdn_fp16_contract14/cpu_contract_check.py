"""CPU-only algebra/shape diagnostics; no device result or acceptance claim."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast, hashlib, json, sys
from pathlib import Path
import torch
assert '+cpu' in torch.__version__
assert not any(k.startswith('torch_npu') for k in sys.modules)
torch.set_num_threads(1)
root=Path('/root/dyq/workspace'); private=Path(__file__).resolve().parent
src=root/'TileOPs/workloads/linear_attention/gdn.py'
tree=ast.parse(src.read_text()); node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gdn_ref')
ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
# Deterministic diagnostic fixture, NOT a replay of the NPU RNG/seed.
# Same shape/dtype as decode-b1-float16; no manifest inputs or tolerances changed.
shape=(1,1,16,128)
a=torch.arange(2048,dtype=torch.float32).reshape(shape)
q=((a%31-15)/128).half();k=((a%23-11)/128).half();v=((a%19-9)/128).half()
g=-(torch.arange(16).reshape(1,1,16).float()/32).half(); beta=(torch.arange(16).reshape(1,1,16).float()/64).half()
h=((torch.arange(16*128*128).reshape(1,16,128,128)%29-14)/2048).float()
o,final=ns['gdn_ref'](q,k,v,g,beta,h)
# KDA per-K-channel decay reduces to scalar GDN when all K entries equal g.
decay=g.float()[...,None].expand(shape).exp()[:,0]
qh=q[:,0].float()*128**-.5;kh=k[:,0].float();vh=v[:,0].float()
decayed=h*decay[...,None]
u=beta[:,0].float()[...,None]*(vh-torch.einsum('nhkv,nhk->nhv',decayed,kh))
state=decayed+kh[...,None]*u[...,None,:]
out=torch.einsum('nhk,nhkv->nhv',qh,state)[:,None].half()
transpose_o,transpose_s=ns['gdn_ref'](q,k,v,g,beta,h.transpose(-1,-2).contiguous(),state_v_first=True)
assert torch.equal(transpose_o,o) and torch.equal(transpose_s.transpose(-1,-2),final)
assert torch.equal(g.float().half(),g) and torch.equal(beta.float().half(),beta)
ids={r['case_id'] for r in json.loads((private/'SCOPE.json').read_text())}
cs=json.loads((private.parent/'linear_deployment_recovery/ALL_CONTRACTS.json').read_text());rows=[]
for c in cs:
 if c['case_id'] not in ids:continue
 p=c['params'];B,S,H,K=p['q'][0];HV,V=p['v'][0][-2:];N=B if 'cu_seqlens' not in p else p['cu_seqlens'][0][0]-1
 assert not any(p[f] for f in ['use_gate_in_kernel','use_beta_sigmoid_in_kernel','use_qk_l2norm_in_kernel','allow_neg_eigval'])
 assert HV%H==0 and p['g'][0]==[B,S,HV] and p['beta'][0]==[B,S,HV]
 assert p['final_state']==[[N,HV,V,K] if p['state_v_first'] else [N,HV,K,V],'float32']
 assert p['o']==p['v']
 rows.append(dict(case_id=c['case_id'],shape_contract='matched',group=HV//H,sequences=N,state_v_first=p['state_v_first'],gate_float32_bytes=B*S*HV*K*4,aqk_akk_bytes=2*B*HV*S*64*2,device_test_this_task=False))
assert len(rows)==14
result=dict(torch_version=torch.__version__,fixture='deterministic algebra fixture; not original seed replay',canonical_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),gate_widening_exact=True,beta_widening_exact=True,state_transpose_exact=True,output_max_abs=(out.float()-o.float()).abs().max().item(),state_max_abs=(state-final).abs().max().item(),cases=rows,ascend_libraries_loaded=[l for l in Path('/proc/self/maps').read_text().splitlines() if any(x in l.lower() for x in ['libascend','libacl','torch_npu'])])
assert not result['ascend_libraries_loaded']
print(json.dumps(result,indent=2));(private/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
# Execute only the binding's bounds expression, never its build/import/ACL transport.
binding=(private.parent/'linear_deployment_recovery/package/binding.py').read_text()
bounds_node=next(n.value for n in ast.walk(ast.parse(binding)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bounds' for t in n.targets))
expr=compile(ast.Expression(bounds_node),'binding-bounds-only','eval')
offsets=torch.tensor([0,1,64,1064,4096],dtype=torch.int64)
try:
 eval(expr,{'offsets':offsets,'offsets_cpu':None})
 raise AssertionError('expected None dereference')
except AttributeError as e:
 old_error=str(e)
fixed='None if offsets is None else (offsets if offsets_cpu is None else offsets_cpu).tolist()'
checks=[]
for device_offsets,cpu_offsets in [(None,None),(offsets,None),(offsets,offsets.clone())]:
 got=eval(fixed,{'offsets':device_offsets,'offsets_cpu':cpu_offsets});assert got==(None if device_offsets is None else [0,1,64,1064,4096]);checks.append(got)
metadata=dict(case_id='varlen-ragged-grouped-float16',original_expression=ast.unparse(bounds_node),original_exception=old_error,patched_results=checks,diagnostic='CPU metadata-only; no NPU invoked; no decode numerical repair claim')
(private/'METADATA_CHECK.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))
