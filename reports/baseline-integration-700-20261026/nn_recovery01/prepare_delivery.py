import ast,datetime,difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');R=json.loads((P/'RESULT.json').read_text()); deploy=P/'merge_packet';deploy.mkdir(exist_ok=True);groups={};mappings=[]
for r in R['rows']:
 if r['status']!='verified_unregistered':continue
 d=P/r['attempt'];a=json.loads((d/'ATTRIBUTION.json').read_text()) if (d/'ATTRIBUTION.json').exists() else {}
 if not (a.get('vendor_seen') and a.get('registration_seen') and a.get('device_launch_seen')):continue
 code=(d/'adapter.py').read_text();h=hashlib.sha256(code.encode()).hexdigest();name='integration700_nn_recovery01_'+h[:10];(deploy/(name+'.py')).write_text(code);ast.parse(code)
 groups.setdefault(r['baseline'],{}).setdefault(r['operator'],[]).append((r['case_id'],name))
 mappings.append(dict(key=[r['baseline'],r['operator'],r['case_id']],factory='benchmarks.ascend_all_pairs.'+r['baseline'].replace('-','_')+'.'+name+'.make',sha256=h,checker=r['checker_receipt'],guard=r['guard_receipt'],attribution=str(d/'ATTRIBUTION.json')))
for base,ops in groups.items():
 target=W/'TileOPs/benchmarks/ascend_all_pairs'/base.replace('-','_')/'__init__.py';old=target.read_text();(deploy/(base+'_init.before.py')).write_text(old);add='\n# nn_recovery01: exact keys with guarded full checker receipts.\n'
 for i,(op,keys) in enumerate(ops.items()):
  fn='_nn_recovery01_'+op;add+=f"{fn}_prior = PROVIDERS.get({op!r})\ndef {fn}(case, _prior={fn}_prior):\n"
  for cid,name in keys:add+=f"    if case._call.signature.name == {op!r} and case.id == {cid!r}:\n        from .{name} import make\n        return make(case)\n"
  add+="    if _prior is not None:\n        return _prior(case)\n    from benchmarks.ascend_baselines import UnsupportedCase\n    raise UnsupportedCase('No accepted exact key')\n"+f"PROVIDERS[{op!r}] = {fn}\n"
 ast.parse(old+add);(deploy/(base+'_registration.append.py')).write_text(add);(deploy/(base+'_registration.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),(old+add).splitlines(True),fromfile=str(target),tofile=str(target))))
(P/'MERGE_MAP.json').write_text(json.dumps(mappings,indent=2));print('merge eligible',len(mappings))

bykey={tuple(x['key']):x for x in mappings}
for r in R['rows']:
 key=(r['baseline'],r['operator'],r['case_id'])
 if key in bykey:
  m=bykey[key];r['factory']=m['factory'];r['reason']='Full canonical checker passed; original vendor, kernel registration and native device launch attribution recorded. Parent merge pending.';r['evidence'].append(m['attribution']);r['changed_files']=[str(deploy/(m['factory'].split('.')[-2]+'.py'))];r['source_hash']=m['sha256']
q=P/'RESULT.json.tmp';q.write_text(json.dumps(R,indent=2));q.replace(P/'RESULT.json')
