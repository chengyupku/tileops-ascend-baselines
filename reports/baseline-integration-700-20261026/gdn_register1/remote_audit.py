import sys,json,pathlib,hashlib,shutil,subprocess,difflib,datetime
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/gdn_register1';A=D/'attempt_001';A.mkdir(parents=True,exist_ok=False)
T=W/'reports/baseline-integration-700-20261026/transformer_linear';cid='q35-397b-tp8-1024-7168-bfloat16';R=T/cid
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,x): (A/n).write_text(json.dumps(x,indent=2)+'\n')
def copy(p):
 p=pathlib.Path(p);q=A/'evidence'/str(p).lstrip('/');q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);return str(q.relative_to(D))
for n in ['STARTED','HEARTBEAT']:(D/n).write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' CPU-only\n')
c=json.loads((R/'checker.json').read_text());g=json.loads((R/'GUARD_RECEIPT.json').read_text())
assert sha(R/'checker.json')=='1acd59c7dacf708ccc434f5f98ed7654d282e0ec7ca88c7072c53badda8167e8'
assert sha(R/'GUARD_RECEIPT.json')=='85f68969c4245d874fc3d50261bdb16e714ea48a376df589cd061b070d3e1dcf'
assert c['status']=='source_changed' and g['returncode']==1 and c['verification']['checked_outputs']==c['verification']['total_outputs']==2 and c['inputs_preserved'] and not c['oracle_source_changed']
for n in ['checker.json','GUARD_RECEIPT.json','child.sh','loaded-libraries.json']:copy(R/n)
b=c['provider_source_sha256'];a=c['provider_source_sha256_after'];changes={p:{'before':b.get(p),'after':a.get(p)} for p in b.keys()|a.keys() if b.get(p)!=a.get(p)}
assert len(changes)==3
snaps={};wanted={h for v in changes.values() for h in v.values() if h}
for team in ['transformer_attention_misc','transformer_linear']:
 for p in (T.parent/team).glob('*.py'):
  h=sha(p)
  if h in wanted:snaps[h]={'path':str(p),'text':p.read_text(),'evidence':copy(p)}
p='TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py';bt=snaps[changes[p]['before']]['text'];at=snaps[changes[p]['after']]['text'];assert at.startswith(bt)
assert 'GDNFwdOp' not in at[len(bt):]
(A/'PROVIDER.diff').write_text(''.join(difflib.unified_diff(bt.splitlines(True),at.splitlines(True))))
save('SOURCE_EQUIVALENCE.json',{'changes':changes,'snapshots':snaps,'gdn_unchanged':True})
checks=[]
for n in ['binding.py','entry.py','direct_acl.py']:
 p=T/'chunk_package'/n;q=T/'frozen_v1/chunk_package'/n
 assert sha(p)==sha(q)
 expected=c['source_sha256'].get(str(p));assert expected is None or sha(q)==expected
 checks.append({'tested':str(p),'frozen':str(q),'sha256':sha(q),'receipt_hash':expected,'evidence':copy(q)})
for rel,h in c['oracle_source_sha256'].items():
 p=W/rel;assert sha(p)==h;checks.append({'path':str(p),'sha256':h,'evidence':copy(p)})
for n in ['CONTRACTS.json','CHUNK_CASES.json']:
 p=T/'frozen_v1'/n;copy(p)
contract=json.loads((T/'frozen_v1/CONTRACTS.json').read_text())[cid];assert contract['params']==c['case_snapshot']['parameters']
pins=json.loads((T/'frozen_v1/chunk_package/RUNTIME_PINS.json').read_text())['files'];recipe=json.loads((T/'frozen_v1/chunk_package/RUNTIME_RECIPE.json').read_text());copy(T/'frozen_v1/chunk_package/RUNTIME_RECIPE.json')
selected=[]
for x in pins:
 p=pathlib.Path(x['path'])
 if ('chunk_gated_delta_rule' in str(p).lower() or str(p) in [recipe['opapi'],recipe['nnopbase']]) and (p.suffix in ['.cpp','.h','.o','.so','.json']):
  h=sha(p);assert h==x['sha256'];item={**x,'current_sha256':h}
  if p.suffix in ['.cpp','.h','.json']:item['evidence']=copy(p)
  if p.suffix in ['.o','.so']:
   item['elf_magic']=p.open('rb').read(4).hex();assert item['elf_magic']=='7f454c46'
   cmd=['readelf','-h',str(p)];r=subprocess.run(cmd,capture_output=True,text=True);item['readelf']={'command':cmd,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
   if str(p)==recipe['opapi']:
    r=subprocess.run(['readelf','-Ws',str(p)],capture_output=True,text=True);item['api_symbols']=[l for l in r.stdout.splitlines() if 'aclnnChunkGatedDeltaRule' in l];assert item['api_symbols']
  selected.append(item)
loaded=json.loads((R/'loaded-libraries.json').read_text());matches=[x for x in loaded if x['path']==recipe['opapi']];assert matches and matches[0]['sha256']==sha(recipe['opapi'])
logs=[]
for p in (R/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
 lines=p.read_text(errors='replace').splitlines();hits=[{'line':i+1,'text':s} for i,s in enumerate(lines) if ('chunkgateddeltarule' in s.lower() or 'chunk_gated_delta_rule' in s.lower()) and any(k in s.lower() for k in ['launch','execute','kernel','binary','workspace'])]
 if hits:logs.append({'path':str(p),'sha256':sha(p),'hits':hits[-50:]})
save('NATIVE.json',{'selected_pins':selected,'loaded_opapi':matches,'launch_logs':logs,'recipe':recipe});save('CHECKS.json',{'critical':checks,'contract':contract,'verification':c['verification'],'checker_status':c['status'],'guard_returncode':g['returncode'],'source_equivalent':True})
for rel in ['TileOPs/benchmarks/ascend_all_pairs/registry.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_linear_gdn.py']:copy(W/rel)
save('READY.json',{'ready':True,'cpu_only':True});print(json.dumps({'ready':True,'pins_checked':len(selected),'launch_logs':len(logs),'evidence_dir':str(A)}))
