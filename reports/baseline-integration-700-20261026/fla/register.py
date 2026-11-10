import json,pathlib,fcntl,shutil,hashlib,ast,datetime
p=pathlib.Path(__file__).resolve().parent
provider=p.parents[2]/'TileOPs/benchmarks/ascend_all_pairs/fla'
assert provider.is_dir(),provider
keys=[]
for f in p.glob('*.checker.json'):
 if f.name=='CumsumFwdOp.checker.json': continue
 d=json.loads(f.read_text());v=d.get('verification',{})
 commands=[c for c in p.glob('*.COMMAND.json') if d['case_id'] in json.loads(c.read_text())]
 if commands:
  command=max(commands,key=lambda c:c.stat().st_mtime); guard=command.with_name(command.name.replace('.COMMAND.json','.GUARD_RESULT.json'))
  if not guard.exists() or json.loads(guard.read_text())['returncode'] in (75,76,78,124):continue
 if d.get('status')=='verified' and v.get('checked_outputs')==v.get('total_outputs') and v.get('checked_outputs',0)>0: keys.append((d['op'],d['case_id']))
existing=set()
for receipt in p.glob('registration.*.json'):existing.update(tuple(k) for k in json.loads(receipt.read_text())['keys'])
keys=sorted(set(keys)-existing)
if not keys:
 print('No new keys to register');raise SystemExit(0)
module='integration700_fla_native'
with open('/tmp/baseline700-provider-fla.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 target=provider/'__init__.py';original=target.read_text();stamp=datetime.datetime.now().strftime('%H%M%S')
 (p/f'provider_init.{stamp}.backup').write_text(original)
 shutil.copyfile(p/(module+'.py'),provider/(module+'.py'))
 text=original+'\n# Exact-key registration baseline700/fla '+stamp+'\n'
 text+='def _integration700_fla_route(previous, exact_ids):\n    def factory(case):\n        if case.id in exact_ids:\n            from .integration700_fla_native import build\n            return build(case)\n        return previous(case)\n    return factory\n'
 for op in sorted(set(k[0] for k in keys)):
  ids=sorted(k[1] for k in keys if k[0]==op)
  text+=f'PROVIDERS[{op!r}] = _integration700_fla_route(PROVIDERS[{op!r}], {ids!r})\n'
 ast.parse(text);target.write_text(text)
 (p/f'registration.{stamp}.json').write_text(json.dumps(dict(keys=keys,module=str(provider/(module+'.py')),sha256=hashlib.sha256((p/(module+'.py')).read_bytes()).hexdigest()),indent=2))
print(keys)
