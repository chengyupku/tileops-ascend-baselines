import ast,builtins,dataclasses,functools,json,sys,types,datetime
from pathlib import Path
from typing import Any, Callable, Optional
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
(O/'HEARTBEAT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='final_contract_review',npu=False)))
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original=builtins.__import__
def safe(name,*a,**k):
 if name.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError(name)
 return original(name,*a,**k)
builtins.__import__=safe
M=types.ModuleType('registration');M.__file__=str(O/'integration700_bf16_grouped_registration1.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
from benchmarks.ascend_baselines import UnsupportedCase
key=next(iter(M.CASE_PARAMS));c=M._frozen_case(key)
assert key==('ops-transformer','GroupedGemmFwdOp','moe-fwd-bfloat16') and len(M.CASE_PARAMS)==1
for name in ('batch_sizes','batch_offsets'):
 specs=dict(c._call.specs);spec=specs[name]
 values=list(spec.values);values[0]+=1;specs[name]=dataclasses.replace(spec,values=values)
 bad=types.SimpleNamespace(id=c.id,params=c.params,_call=dataclasses.replace(c._call,specs=specs))
 assert not M.matches(bad)
 try:M.make(bad)
 except UnsupportedCase:pass
 else:raise AssertionError('changed metadata accepted')
for name in c._call.specs:
 specs=dict(c._call.specs);specs.pop(name)
 bad=types.SimpleNamespace(id=c.id,params=c.params,_call=dataclasses.replace(c._call,specs=specs))
 assert not M.matches(bad)
specs=dict(c._call.specs);specs['a']=dataclasses.replace(specs['a'],cpu=True)
assert not M.matches(types.SimpleNamespace(id=c.id,params=c.params,_call=dataclasses.replace(c._call,specs=specs)))
unknown=types.SimpleNamespace(id='square-4k-bfloat16',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='BmmFwdOp')))
assert M.wrap(lambda _: 'previous-only')(unknown)=='previous-only'
try:M.wrap(None)(unknown)
except UnsupportedCase:pass
else:raise AssertionError('Bmm accepted')
adapter=M._load(M._snapshot())['adapter']
run,_=adapter.make_runner('GroupedGemmFwdOp',c._call.indices,native_factory=lambda _:types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda *a:None)),tensor_api=types.SimpleNamespace(bfloat16='bfloat16',int32='int32'))
assert tuple(__import__('inspect').signature(run).parameters)==('a','b','sizes','offsets')
result=dict(passed=True,checks=['original metadata values, tensor presence and CPU flag mutations fail closed','single exact BF16 NT scope; Bmm only delegates previous and never admitted','original adapter requires exactly four named inputs a,b,sizes,offsets'],npu_runs=0,device_imports=0)
(O/'SUPPLEMENTAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
with (O/'FEEDBACK.txt').open('a') as f:f.write('27 template CPU groups PASS; all 7525 pins hashed once, no device imports/native loads; recipe static checks PASS and not executed. Supplemental exact metadata/presence/four-input/Bmm-exclusion checks completed.\n')
