from pathlib import Path
import json,hashlib,datetime,fcntl,os,shutil,subprocess,sys,yaml
D=Path(__file__).resolve().parent;W=D.parents[1];O=D/'tla_sdk_recovery36';R=D/'tilelang_register5';P=R/'attempt02';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
rows=read(R/'RECOMMENDED_ACCEPT.json')['rows'];assert len(rows)==5;reg=read(P/'REGISTRATION.json');audit=read(P/'AUDIT.json');prov=read(P/'LOCAL_ORIGINAL_PROVENANCE.json');pin=read(P/'pins.json')
assert sha(P/'frozen/impl.py')==audit['impl_sha256']=='fdf4bc496ac2c06b2cb6290680652b77248a95ba17c443561f3eedcc86f77500'
original=P/'frozen/vendor/examples/ascend/example_per_token_cast_to_fp8.py';assert sha(original)==prov['git_blob_sha256']==audit['original_sha256']
git=subprocess.run(['git','-C',prov['repo'],'show',prov['commit']+':'+prov['path']],capture_output=True);assert git.returncode==0 and hashlib.sha256(git.stdout).hexdigest()==sha(original)
assert '@tilelang.jit(out_idx=[1, 2], target="ascend"' in original.read_text();assert 'kernel=m.per_token_cast_simt(rows,128)' in (P/'frozen/impl.py').read_text()
spec=yaml.safe_load((W/'TileOPs/src/tileops/manifest/spec/quantization.yaml').read_text())['FP8QuantFwdOp'];evidence={};sources={};out=[]
for r in rows:
 c=read(r['checker_receipt']);g=read(r['guard_terminal_receipt']);gm=read(r['guard_receipt']);run=Path(r['checker_receipt']).parent;child=read(run/'guard.json');key=[r['baseline'],r['operator'],r['case_id']]
 assert [c['library'],c['op'],c['case_id']]==key
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed'];v=c['verification'];assert v['checked_outputs']==v['total_outputs']==2 and v['unchecked_reason'] is None
 assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big')
 assert g['returncode']==child['returncode']==0 and gm['owner']=='baseline700-20261026/tla_sdk_recovery36' and gm['phase']=='running'
 assert g['argv'][:2]==['npu-guard','run'] and g['argv'][g['argv'].index('--owner')+1]==gm['owner']
 assert c['edge_check']['changed_inputs'] and c['edge_check']['zero_groups'] and c['edge_check']['tiny_groups'] and c['edge_check']['scale_floor'] and c['edge_check']['independent_cpu_reference']
 assert all(x['returned'] for x in read(run/'launch.json'))
 work=next(x for x in spec['workloads'] if key[2].startswith(x['label']+'-'));dtype=key[2].rsplit('-',1)[1];shape=[work[n] for n in ['batch','seq_len_kv','kv_group','index_dim']]
 obs=c['observed_calls'];a=obs['candidate'][0];ref=obs['reference'][0];assert a['inputs']==ref['inputs'] and a['returned'] and ref['returned']
 assert a['inputs']['items'][0]['shape']==shape and a['inputs']['items'][0]['dtype']==dtype
 outputs=a['outputs']['items'];assert outputs[0]['shape']==shape[:-1] and outputs[1]['shape']==shape;assert outputs[0]['dtype']=='float32' and outputs[1]['dtype']=='float8_e4m3fn'
 for p,h in c['source_sha256'].items():sources[p]=h
 for p,h in c['oracle_source_sha256'].items():assert sha(W/p)==h
 assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py')
 for p in [r['checker_receipt'],r['guard_receipt'],r['guard_terminal_receipt'],run/'guard.json',run/'launch.json']:evidence[str(p)]=sha(p)
 # Cross-check the original preserved receipt rather than only the registrar copy.
 assert sha(O/'runs'/run.name/'checker.json')==sha(r['checker_receipt'])
 out.append({'key':key,'checker':{'path':r['checker_receipt'],'sha256':sha(r['checker_receipt'])},'guard':{'path':r['guard_terminal_receipt'],'sha256':sha(r['guard_terminal_receipt']),'exit_code':0},'verification':v,'seed':c['input_seed'],'inputs_preserved':True,'canonical':work,'edge_check':c['edge_check'],'formal_registration':{'factory':r['factory'],'module_sha256':reg['module_sha256'],'frozen_impl_sha256':audit['impl_sha256'],'original_source_sha256':audit['original_sha256']},'runtime_recipe':str(R/'run_registered.py')})
native=[]
for m in (P/'native').rglob('manifest.json'):
 for name,v in read(m)['files'].items():assert sha(m.parent/name)==v['sha256']
 native.append({'manifest':str(m),'sha256':sha(m)})
for m in (P/'native').rglob('metadata.json'):
 j=read(m);binary=m.parent/'kernel.aibin';assert sha(binary)==j['sha256'] and binary.read_bytes()[:4]==b'\x7fELF';native.append({'binary':str(binary),'sha256':sha(binary)})
assert len(native)==6
# Read-only live hash and real factory construction until the intentionally blocked torch import.
code=(R/'cpu_registry.py').read_text();code=code.replace("(P/'attempt02/REGISTRY_CPU.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))","print(json.dumps(result,indent=2))")
extra='import hashlib,json,pathlib\nchecks='+repr(sources)+'\nfor p,h in checks.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p\n'
module='/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/tilelang/integration700_tilelang_register5.py';extra+='assert hashlib.sha256(pathlib.Path('+repr(module)+').read_bytes()).hexdigest()=='+repr(reg['module_sha256'])+'\n'
p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495','python3 -S -'],input="import os;os.chdir('/opt/tileops-all-pairs/workspace')\n"+extra+code,text=True,capture_output=True,timeout=50);assert p.returncode==0,p.stderr
proof=json.loads(p.stdout);assert proof['no_device_import'] and len(proof['rows'])==5 and proof['outside_scope_rejected'];put(D/'PARENT_TILELANG5_LIVE_CHECK.json',proof)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rp=D/'PARENT_TILELANG5_REVIEW.json';review={'at':now,'scope':5,'full_outputs_checked':10,'receipt_hashes':evidence,'native_cached_artifacts':native,'tested_source_hashes_verified':len(sources),'git_original_provenance':prov,'registry_keys':5,'new_device_runs':0,'boundary':'Complete original per_token_cast_simt with unchanged target=ascend/pass config/kernel. Exact FP32 widening, zero-pad64 to128, scale/code output reorder and shape trim only. SDK920b2 supplies original compiler/include environment; no compiler/header/kernel source edits.','numeric_contract':'Unchanged FP8 verifier allows adjacent code difference; finite outputs/scales tolerance and full zero/tiny scale-floor probes passed. Raw max_abs32 is not a bitwise comparison requirement.','scope_remaining':31,'live_check':str(D/'PARENT_TILELANG5_LIVE_CHECK.json'),'timings':read(O/'DELIVERY.json')['timings'],'limitations':'Runtime success applies to five exact FP8 keys only. Retained binary/cache receipts are original compiled artifacts, not new launch claims. Three other representative failures preserve their separate causes.'}
put(rp,review)
LP=D/'ACCEPTED.json'
with open(str(LP)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(LP);by={tuple(x['key']):x for x in ledger['rows']};before=len(by)
 for r in out:
  if tuple(r['key']) not in by:r.update(accepted=True,at=now,parent_review=str(rp));by[tuple(r['key'])]=r
 ledger.update(rows=list(by.values()),updated_at=now);put(LP,ledger)
review.update(accepted_added=len(by)-before,accepted_total=len(by));put(rp,review);put(O/'PARENT_REVIEW.json',{'at':now,'scope':36,'accepted':5,'remaining':31,'review':str(rp)});put(R/'PARENT_REVIEW.json',{'at':now,'accepted':5,'review':str(rp)})
s=read(D/'STATE.json');s.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(rp));put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as a,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_tilelang5_'+now.replace(':','-')+'.json'));j=read(M);j['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(rp),last_status_at=now);put(M,j)
print(json.dumps({'added':len(by)-before,'accepted':len(by),'outputs':10,'source_hashes':len(sources),'native_artifacts':len(native)}))
