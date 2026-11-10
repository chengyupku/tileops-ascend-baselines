import json,hashlib,pathlib,shutil,re,datetime
O=pathlib.Path(__file__).resolve().parent; P=O.parent/'threehost_linear49'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
entries=read(P/'REGISTRATION_HANDOFF.json')['entries']; scope=read(O/'SCOPE.json')
assert {tuple(x['exactkey']) for x in entries}=={(x['baseline'],x['operator'],x['case_id']) for x in scope}
pins={};audit=[];local={}; evidence={}
for e in entries:
 cid=e['exactkey'][2];D=(P/e['checker']).parent;c=read(D/'checker.json');a=read(P/e['audit']);g=read(D/'GUARD_RESULT.json');loaded=read(D/'LOADED_HASHES.json');package=P/e['provider_package'];remote=e['provider_package'].startswith('server1/')
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==2 and g['returncode']==0
 assert sha(D/'checker.json')==a['checker_sha256']
 assert all(x['returned'] and len(x['outputs']['items'])==2 for group in c['observed_calls'].values() for x in group)
 frozen=O/'closure'/e['provider_package'];frozen.mkdir(parents=True,exist_ok=True)
 for name in ('binding.py','direct_acl.py','entry.py','RUNTIME_RECIPE.json','RUNTIME_PINS.json'):
  src=package/name; dst=frozen/name;shutil.copyfile(src,dst);local[str(dst.relative_to(O))]=sha(dst)
  if name in ('binding.py','direct_acl.py'):
   runtime=str(src).replace(str(P/'server1'),'/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_linear49') if remote else str(src)
   assert loaded[runtime]['sha256']==sha(src),(cid,name)
 for name in ('CONTRACTS.json','CHUNK_CASES.json'):
  dst=frozen.parent/name;shutil.copyfile(P/name,dst);local[str(dst.relative_to(O))]=sha(dst)
 freeze_name=a['native_pin_manifest'];native=read(P/freeze_name)['files'];runtime={x['path']:x['sha256'] for x in native}
 # Pre-execution native pins are linked to execution by pointer/loaded hash and exact registered device object.
 recipe=read(package/'RUNTIME_RECIPE.json');ds=[json.loads(x) for x in (D/'dispatch.jsonl').read_text().splitlines()];maps=(D/'maps.txt').read_text().splitlines();ptr=[]
 for d in ds:
  assert d['library']==recipe['opapi']
  for field in ('symbol','workspace_symbol'):
   address=int(d[field],16);m=next(line.split(maxsplit=5) for line in maps if int(line.split()[0].split('-')[0],16)<=address<int(line.split()[0].split('-')[1],16))
   assert m[5]==d['library'] and int(m[4])==d['inode']
   assert loaded[d['library']]['inode']==d['inode'] and loaded[d['library']]['device']==d['device']
   assert loaded[d['library']]['sha256']==runtime[d['library']]
   ptr.append({'pointer':field,'library':d['library'],'inode':d['inode'],'maps_device':m[3],'stat_device':d['device']})
 log=(D/'NATIVE_DISPATCH_EXTRACT.log').read_text();bins=re.findall(r'bin path: (\S+)',log)
 assert bins and 'StreamLaunchKernel' in log
 for binary in bins:assert binary in runtime,binary
 nn=[k for k in loaded if k.endswith('/libnnopbase.so')];assert len(nn)==1
 assert ('cann-9.1.0' if remote else 'cann-9.2.0-beta.2') in nn[0]
 # Pin the actual loaded native runtime and Python providers, not an unbounded system tree.
 for name,item in loaded.items():
  if ('/Ascend/' in name or '/torch/' in name or '/torch_npu/' in name or '/fla/' in name or '/triton/' in name) and (' (deleted)' not in name):
   if name in runtime:assert runtime[name]==item['sha256'],name
   runtime[name]=item['sha256']
 for name,h in c['oracle_source_sha256'].items():
  matches=[k for k,v in loaded.items() if k.endswith('/'+name) and v['sha256']==h] if not name.startswith('/') else [name]
  for path in matches:runtime[path]=h
 env=read(D/'ENV.json');env.pop('GDN_CASE_OWNER',None)
 env['ASCEND_CUSTOM_OPP_PATH']=str(pathlib.PurePosixPath(recipe['opapi']).parents[2])
 pins[cid]={'host':c['runtime_host'],'package':str(frozen.relative_to(O)), 'recipe':recipe,'environment':env,'runtime':runtime,'native_pin_receipt':freeze_name,'registered_device_objects':{n:runtime[n] for n in bins},'python_prefix':'/opt/tileops-ci/runtime' if remote else '/root/dyq/workspace/.venv'}
 receipt_names=['checker.json','GUARD_RESULT.json','GUARD_RUNNING.json','VALIDATION_AUDIT.json','dispatch.jsonl','maps.txt','LOADED_HASHES.json','NATIVE_DISPATCH_EXTRACT.log','ENV.json']
 for n in receipt_names:evidence[str((D/n).relative_to(P))]=sha(D/n)
 audit.append({'exactkey':e['exactkey'],'architecture':c['runtime_host'],'checker_sha256':sha(D/'checker.json'),'checked_outputs':2,'inputs_preserved':True,'guard_returncode':0,'input_seed':c['input_seed'],'pointer_checks':ptr,'registered_device_objects':pins[cid]['registered_device_objects'],'native_pins':freeze_name,'binding_matches_execution':True,'native_full_source_claim':False,'checker':e['checker']})
# Exact original passive API is extracted at runtime, never importing torch at construction.
for src,rel in [(P/'snapshot/TileOPs/benchmarks/api.py','closure/api.py'),(P/'snapshot/TileOPs/scripts/ascend_pair_check.py','closure/evidence/ascend_pair_check.py'),(P/'snapshot/TileOPs/benchmarks/ascend_delta_reference.py','closure/evidence/ascend_delta_reference.py'),(P/'snapshot/TileOPs/workloads/linear_attention/deltanet.py','closure/evidence/deltanet.py')]:
 dst=O/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);local[rel]=sha(dst)
# Freeze only relevant original API/kernel sources, with provenance distinguished from runtime attestation.
for subtree in ('recurrent_kda','chunk_gated_delta_rule'):
 for src in (P/'original_source'/subtree).rglob('*'):
  if src.is_file():
   dst=O/'closure/original_source'/subtree/src.relative_to(P/'original_source'/subtree);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);local[str(dst.relative_to(O))]=sha(dst)
write(O/'RUNTIME_PINS.json',pins);write(O/'LOCAL_PINS.json',local);write(O/'AUDIT.json',audit);write(O/'EVIDENCE_HASHES.json',evidence)
write(O/'HEARTBEAT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'frozen_execution_chain_8_pass','npu_used':False})
print(json.dumps({'audit_passed':len(audit),'frozen_files':len(local),'runtime_pin_counts':{k:len(v['runtime']) for k,v in pins.items()}}))
