from pathlib import Path
import json,hashlib,subprocess,datetime,fcntl,os,shutil
D=Path(__file__).resolve().parent;W=D.parents[1];R=D/'audit_dense33'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
rows=[r for r in read(R/'RECOMMENDED_ACCEPT.json')['rows'] if r['decision']=='recommended_accept'];assert len(rows)==28
local={};remote={};native=[];original={}
def refs(o):
 if isinstance(o,dict):
  p=o.get('path');h=o.get('sha256')
  if p and h:
   if p.startswith(str(W)+'/'):assert sha(p)==h,p;local[p]=h
   elif p.startswith('/opt/'):remote[p]=h
  for v in o.values():refs(v)
 elif isinstance(o,list):
  for v in o:refs(v)
for r in rows:
 assert r['accept_recommended'] and not r['issues'];refs(r);c=read(r['checker']['path']);v=c['verification'];key=r['key']
 assert [c['library'],c['op'],c['case_id']]==key
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert v['checked_outputs']==v['total_outputs']>0 and v['unchecked_reason'] is None
 assert c['input_seed']==r['seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
 assert c['case_snapshot']==r['case_snapshot']==r['canonical']['shape_dtype_layout']
 assert int(Path(r['guard']['exit']['path']).read_text())==r['guard']['returncode'] in (0,1)
 if r['guard']['returncode']==1:
  assert key[1]=='YaRNFwdOp';later=read(r['guard']['later_failure']['checker']['path']);assert later['started_at']>c['finished_at'] and later['status']=='failed' and 'NoneType' in later['error']
 for p,h in c['oracle_source_sha256'].items():assert sha(W/p)==h,p
 s=r['source_provenance'];assert not s['drift'];remote.update(s['primary_hashes']);remote.update(s['kernel_hashes']);remote.update(s['logs'])
 remote[r['checker_remote']['path'] if isinstance(r['checker_remote'],dict) else r['checker_remote']]=r['checker']['sha256']
 remote[r['guard']['remote_path']]=r['guard']['sha256']
 n=s['native_per_case'];assert n['evidence'];native.extend(n['evidence'])
 for e in n['evidence']:
  remote[e['file']]=e['log_sha256']
  import re
  tm=re.search(r'(2026-10-26-\d\d:\d\d:\d\d)\.(\d+)(?:\.(\d+))?',e['text']);assert tm
  t=tm[1].replace('2026-10-26-','2026-10-26T')+'.'+tm[2]+(tm[3] or '')+'+00:00'
  assert datetime.datetime.fromisoformat(c['started_at'])<=datetime.datetime.fromisoformat(t)<=datetime.datetime.fromisoformat(c['finished_at']),(key,t)
 if key[1]!='YaRNFwdOp':assert r['tested_registered_identity']['identical'] and r['tested_registered_identity']['tested_sha256']==r['formal_module']['sha256']
 for x in r['critical_dependencies']['checks']:assert x['matches'] and x['expected']==x['sha256'];remote[x['path']]=x['sha256']
keys={tuple(r['key']) for r in rows}
for r in read(R/'ORIGINAL_SOURCE_CHECK.json'):
 if tuple(r['key']) not in keys:continue
 assert not r['missing'] and not r['different']
 for f in r['files']:
  assert f['matches'] and sha(f['path'])==f['sha256'];original[f['path']]=f['sha256'];remote[f['installed']]=f['sha256']
src=(W/'TileOPs/scripts/ascend_pair_check.py').read_text();end=src.index('args.output.write_text',src.index('    finally:'));assert src.index('torch.npu.synchronize()',src.index('    finally:'))<end and src.index('candidate.cleanup()',src.index('    finally:'))<end
scope=[r for r in read(R/'SCOPE.json') if (r['baseline'],r['operator'],r['case_id']) in keys]
# Reviewed CPU-only script uses fake device constructors and never invokes run().
code=(R/'remote_extra.py').read_text().replace('SCOPE_PLACEHOLDER',repr(scope))
prefix='''import pathlib,hashlib,sys,fcntl
sys.dont_write_bytecode=True
class BlockNPU:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl'):raise RuntimeError('forbidden device import '+fullname)
sys.meta_path.insert(0,BlockNPU())
hashes=HASHES
for p,h in hashes.items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
native=NATIVE
for e in native:assert e['text'] in pathlib.Path(e['file']).read_text(errors='replace'),e
lock=open('/tmp/baseline700-provider-ops_transformer.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
'''.replace('hashes=HASHES','hashes='+repr(remote)).replace('native=NATIVE','native='+repr(native))
p=subprocess.run([str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'),'openlibing-DevEnv-661495','python3 -S -'],input=prefix+code,text=True,capture_output=True,timeout=55)
(D/'PARENT_DENSE28_LIVE_CHECK.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
proof=json.loads(p.stdout);assert len(proof['routes'])==28 and proof['yarn']['identity_reconstructed'] and proof['yarn']['elf_sections_equal_except']==['.strtab'];put(D/'PARENT_DENSE28_LIVE_CHECK.json',proof)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rp=D/'PARENT_DENSE28_REVIEW.json'
review={'at':now,'scope':28,'local_receipt_hashes':len(local),'remote_critical_hashes':len(remote),'original_source_files_matched':len(original),'native_interval_records':len(native),'actual_factory_routes':28,'npu_runs':0,'boundary':'Original full GroupedMatmulV5 INT8/FP8 API, independent BMM calls and grouped packing, unchanged original MoEInitRoutingV2 with exact widening/all3 outputs, original rotary operator with canonical frequency state. No K splitting, kernel edits, fallback or oracle changes.','qualification':'YaRN retains true enclosing guard1 from later LongRoPE None.cpu error. Its full verified receipt follows device sync and cleanup. Tested/current YaRN binding equivalence reconstructed; two original native ELFs match all sections except generated filename string table. No direct per-case launch path in verbosity3 log; source/binary equivalence supplies ownership.','held_keys':[r['key'] for r in read(R/'RECOMMENDED_ACCEPT.json')['rows'] if r['decision']!='recommended_accept'],'live_evidence':str(D/'PARENT_DENSE28_LIVE_CHECK.json')};put(rp,review)
with open(str(D/'ACCEPTED.json')+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(D/'ACCEPTED.json');by={tuple(r['key']):r for r in ledger['rows']};before=len(by)
 for r in rows:
  if tuple(r['key']) not in by:by[tuple(r['key'])]={**r,'accepted':True,'at':now,'parent_review':str(rp)}
 ledger.update(rows=list(by.values()),updated_at=now);put(D/'ACCEPTED.json',ledger)
review.update(accepted_added=len(by)-before,accepted_total=len(by));put(rp,review);put(R/'PARENT_REVIEW.json',review)
s=read(D/'STATE.json');s.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(rp));put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as a,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_dense28_'+now.replace(':','-')+'.json'));j=read(M);j['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(rp),last_status_at=now,gmm_reference_review=str(D/'gmm_reference_audit49/PARENT_REVIEW.json'));put(M,j)
print(json.dumps({k:review[k] for k in ['accepted_added','accepted_total','local_receipt_hashes','remote_critical_hashes','original_source_files_matched','native_interval_records']}))
