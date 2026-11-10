from pathlib import Path
import fcntl,shutil,hashlib,json,ast,datetime
p=Path(__file__).parent;d=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer')
items=[('MoEPrePermuteFwdOp',['decode-t32-bfloat16','decode-t32-float16'],'pre','pre','pre_check_graph/decode-t32-bfloat16.json')]
for op,ids,mod,func,receipt in items:
 assert json.loads((p/receipt).read_text())['status']=='verified'

assert json.loads((p/'pre_check_graph/decode-t32-float16.json').read_text())['status']=='verified'
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);src=(d/'__init__.py').read_text();shutil.copy2(d/'__init__.py',p/'provider_init.pre.before.py');block='\n# integration700_transformer_dense graph exact verified keys\n';rows=[]
 for op,ids,mod,func,receipt in items:
  tag='_integration700_transformer_dense_'+op
  assert tag not in src
  block+=f'from .integration700_transformer_dense_{mod} import {func} as {tag}_factory\n{tag}_old = PROVIDERS.get({op!r})\ndef {tag}_dispatch(case):\n    if case.id in {ids!r}: return {tag}_factory(case)\n    if {tag}_old is not None: return {tag}_old(case)\n    from benchmarks.ascend_baselines import UnsupportedCase\n    raise UnsupportedCase("No exact integration700 key")\nPROVIDERS[{op!r}] = {tag}_dispatch\n'
  name=f'integration700_transformer_dense_{mod}.py';target=d/name
  if target.exists():assert target.read_bytes()==(p/name).read_bytes()
  else:shutil.copy2(p/name,target)
  for cid in ids:rows.append(dict(key=['ops-transformer',op,cid],factory=f'benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_dense_{mod}.{func}',module_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),registered=True))
 new=src+block;ast.parse(new);tmp=d/'integration700_transformer_dense_init.tmp';tmp.write_text(new);tmp.replace(d/'__init__.py');(p/'provider_init.pre.patch').write_text(block);(p/'REGISTRATION_PRE.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,init_before_sha256=hashlib.sha256(src.encode()).hexdigest(),init_after_sha256=hashlib.sha256(new.encode()).hexdigest(),lock='/tmp/baseline700-provider-ops_transformer.lock'),indent=2))
print('registered pre exact keys')
