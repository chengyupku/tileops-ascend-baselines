from pathlib import Path
import json,hashlib,ast,datetime,difflib
O=Path(__file__).resolve().parent;E=O.parent/'threehost_attention_moe53/remote_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
# Only one metadata implementation changed; the complete numerical run AST is identical.
a=ast.parse((E/'dsa.py').read_text());b=ast.parse((E/'dsa.original.py').read_text())
run=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='run')
assert ast.dump(run(a))==ast.dump(run(b))
expected=(E/'dsa.original.py').read_text().replace('from benchmarks.ascend_baselines import Candidate,source_hashes','from benchmarks.ascend_baselines import Candidate\n    import hashlib\n    def source_hashes(*paths):\n        return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}')
assert expected==(E/'dsa.py').read_text()
mm=json.loads((O/'MMAP_CURRENT.json').read_text());mount=json.loads((O/'MOUNT_CURRENT.json').read_text())
assert mm['sha256']=='a695a324f0809ca6948eb86531d889bd330f64b6308ea682053b008a8bdd0e1d'
assert mm['inode']==21475129863 and mm['stat_device']==59 and 'fd:03' in mm['maps'][0]
assert ' 0:59 / / ' in mount['mountinfo'] and ' 253:3 ' in mount['mountinfo'] and ' - overlay ' in mount['mountinfo']
rows=[]
for n,attempt in [('96','06'),('128','01')]:
 case=f'mla-d576-h128-kv{n}k-bfloat16';p=E/(case+'-attempt'+attempt)
 c=json.loads((p/'checker.json').read_text());g=json.loads((p/'guard.json').read_text());native=json.loads((p/'native.json').read_text());d=json.loads((E/f'DSA{n}_DISPATCH_AND_BINARIES.json').read_text())
 assert g['exit_code']==0 and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 call=native['calls'][0];assert call['synchronized'] and call['inputs']==19 and call['outputs']==4
 mappings=[]
 for line in (p/'maps.txt').read_text().splitlines():
  fields=line.split(maxsplit=5)
  lo,hi=(int(v,16) for v in fields[0].split('-'))
  if lo<=call['pointer']<hi:
   path=fields[5];pin=native['mapped_files'][path];assert path.endswith('/libacl_op_compiler.so') and int(fields[4])==pin['inode']==mm['inode'] and pin['sha256']==mm['sha256'];mappings.append(line)
 assert len(mappings)==1
 artifacts=[]
 for path,pin in d['artifacts'].items():
  local=E/path.split('/threehost_attention_moe53/',1)[1];assert sha(local)==pin['sha256'] and local.stat().st_size==pin['bytes']
  artifacts.append({'path':path,**pin})
 binary=next(x for x in artifacts if x['path'].endswith('.o'));name=Path(binary['path']).stem
 launch=[x for x in d['dispatch'] if 'StreamLaunchKernelV1:' in x['text'] and name in x['text']];assert len(launch)==1
 # Validate extracted launch line against complete raw log, not just an assertion summary.
 for x in launch:
  raw=E/x['file'].split('/threehost_attention_moe53/',1)[1];assert raw.read_text(errors='replace').splitlines()[x['line']-1]==x['text']
 rows.append(dict(case_id=case,checker_status=c['status'],verification=c['verification'],input_seed=c['input_seed'],verifier_sha256=c['verifier_sha256'],oracle_source_sha256=c['oracle_source_sha256'],provider_source_sha256=c['provider_source_sha256'],started_at=c['started_at'],finished_at=c['finished_at'],wall_seconds=c['wall_seconds'],runtime_host=c['runtime_host'],snapshot=c['case_snapshot'],native_call=call,pointer_map=mappings,artifacts=artifacts,launch=launch,guard_exit_code=g['exit_code'],guard_scope=g['coordination_scope']))
save('EVIDENCE_REVIEW.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,dsa_difference='Only source_hashes metadata implementation; exact complete run AST and all other bytes unchanged.',current_mount_collection=mount['utc'],current_mmap_collection=mm['utc'],historical_limit='No historical mountinfo was collected with either successful checker. Current same-boot server2 mountinfo plus read-only mmap reproduces stat 0:59/maps fd:03 with identical inode and bytes; this is subsequent corroboration, not a historical mount snapshot.',accepted_added=0))
print('2 checker/guard/native/dispatch reviews passed; metadata-only DSA change; current mount correlation passed')
