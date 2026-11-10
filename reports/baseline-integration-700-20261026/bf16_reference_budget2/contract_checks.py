from pathlib import Path
import json,sys
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T))
s=(T.parent/'bf16_bmm_conv_grouped_prep10/cpu_checks.py').read_text().split('scope=json.loads')[0].replace('import yaml','').replace("bfloat16='bfloat16'","bfloat16='bfloat16',float16='bfloat16'")
exec(compile(s,'mock_definitions','exec'),globals())
results=[]
for r in json.loads((T/'CASES.json').read_text()):
 p=r['manifest_params'];op=r['key'][1];n=Native();run,_=adapter.make_runner(op,p,native_factory=n.factory,tensor_api=api)
 if op=='BmmFwdOp':
  args=(Tensor((p['B'],p['M'],p['K']),'bfloat16'),Tensor((p['B'],p['K'],p['N']),'bfloat16'));out=run(*args);assert len(n.calls)==1 and n.calls[0][:2]==args
 else:
  ns=[p['M']//p['G']]*p['G'];offsets=[sum(ns[:i]) for i in range(p['G'])]
  args=(Tensor((p['M'],p['K']),'bfloat16'),Tensor((p['G'],p['N'],p['K']) if p['transpose_b'] else (p['G'],p['K'],p['N']),'bfloat16'),Tensor((p['G'],),'int32',values=ns),Tensor((p['G'],),'int32',values=offsets))
  out=run(*args);assert len(n.calls)==p['G'] and len(out.writes)==p['G']
  for i,(lhs,rhs,_) in enumerate(n.calls):
   assert ('index',i) in rhs.trace and (('permute',(1,0)) in rhs.trace)==p['transpose_b'];assert ('slice',offsets[i],offsets[i]+ns[i]) in lhs.trace
  rejects(lambda:run(args[0],args[1],args[2],Tensor((p['G'],),'int32',values=[0]*p['G'])))
 assert out.dtype=='bfloat16';assert not any(x.writes for x in args)
 results.append(dict(key=r['key'],calls=len(n.calls),shape=out.shape,dtype=out.dtype))
assert 'torch' not in sys.modules
(T/'CONTRACT_CHECKS.json').write_text(json.dumps(dict(status='pass',rows=results,numerical=False),indent=2))
print(results)
