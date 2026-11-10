"""Publish only complete numerical passes, under the shared provider lock."""
import pathlib,json,fcntl,hashlib,shutil,difflib
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];provider=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'; rows=[]
for f in (P/'runs').glob('*/checker.json'):
 d=json.load(open(f));g=f.parent/'guard.json'
 if d.get('library')!='ops-transformer':continue
 if not g.exists():
  for batch in P.glob('*batch.json'):
   if any(item[0]==d['case_id'] for item in json.load(open(batch))):g=P/'runs'/batch.name/'guard.json';break
 accepted_teardown = g == P/'runs/dsa_small_batch.json/guard.json' and (P/'TEARDOWN_TIMEOUT_EVIDENCE.json').exists()
 if d.get('status')=='verified' and g.exists() and (json.load(open(g))['returncode']==0 or accepted_teardown):
  assert d['verification']['checked_outputs']==d['verification']['total_outputs']
  rows.append((d['op'],d['case_id']))
assert rows
byop={}
for op,cid in rows:byop.setdefault(op,[]).append(cid)
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 init=provider/'__init__.py';before=init.read_text();digest=hashlib.sha256(before.encode()).hexdigest();(P/('init-before-'+digest+'.py')).write_text(before)
 modules=['integration700_transformer_attention_misc_dense.py','integration700_transformer_attention_misc_decode.py']
 if 'DSADecodeWithKVCacheFwdOp' in byop:modules.append('integration700_transformer_attention_misc_dsa.py')
 for n in modules:
  shutil.copyfile(P/n,provider/n)
 code='\n# integration700 transformer_attention_misc: complete canonical passes only\n'
 for op,cids in byop.items():
  module='integration700_transformer_attention_misc_dense' if op=='GQADenseFwdOp' else ('integration700_transformer_attention_misc_dsa' if op=='DSADecodeWithKVCacheFwdOp' else 'integration700_transformer_attention_misc_decode');fn='gqa' if op=='GQADenseFwdOp' else 'factory';tag='_i700_misc_'+op
  code+=f'from .{module} import {fn} as {tag}\n'
  code+=f'def {tag}_route(case, previous=PROVIDERS.get({op!r})):\n    if case.id in {set(cids)!r}:\n        return {tag}(case)\n    if previous is not None:\n        return previous(case)\n    from benchmarks.ascend_baselines import UnsupportedCase\n    raise UnsupportedCase("No registered factory for this exact key")\nPROVIDERS[{op!r}] = {tag}_route\n'
 after=before+code;compile(after,str(init),'exec');init.write_text(after)
 patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(init),tofile=str(init)))
 (P/'provider.patch').write_text(patch)
 (P/('provider-'+hashlib.sha256(after.encode()).hexdigest()+'.patch')).write_text(patch)
 (P/'REGISTRATION.json').write_text(json.dumps({'keys':rows,'before_sha256':digest,'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'modules':{n:hashlib.sha256((provider/n).read_bytes()).hexdigest() for n in modules}},indent=2))
print(rows)
