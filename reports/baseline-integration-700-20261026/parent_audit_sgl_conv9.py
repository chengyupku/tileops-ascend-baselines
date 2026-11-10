from pathlib import Path
import json,hashlib,subprocess,datetime,fcntl,os,shutil,re,sys,yaml
D=Path(__file__).resolve().parent;W=D.parents[1];O=D/'sgl_conv_sampling02';P=D/'sgl_conv_register9'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
reg=read(P/'REGISTRATION.json');rs=reg['rows'];assert len(rs)==9
module=W/reg['module'];assert sha(module)==reg['adapter_hash']==sha(O/'integration700_sgl_conv_sampling02.py')
assert (W/reg['provider']).read_bytes().startswith((P/'provider_init.before').read_bytes())
# Exercise the live registry and real registered factory constructors with device imports stubbed.
code=(P/'cpu_check.py').read_text().replace("pkg=importlib.import_module('benchmarks.ascend_all_pairs.sgl')","pkg=importlib.import_module('benchmarks.ascend_all_pairs.sgl'); registry=importlib.import_module('benchmarks.ascend_all_pairs.registry')")
code=code.replace("self._call=types.SimpleNamespace(op=op)","self._call=types.SimpleNamespace(op=op,signature=types.SimpleNamespace(name=op))")
code=code.replace("pkg.PROVIDERS['Conv2dFwdOp'](C('Conv2dFwdOp',cid))","registry.candidate_factory(C('Conv2dFwdOp',cid),'sgl-kernel-npu')()")
code=code.replace("pkg.PROVIDERS['ChainSpeculativeSamplingFwdOp'](C('ChainSpeculativeSamplingFwdOp',cid))","registry.candidate_factory(C('ChainSpeculativeSamplingFwdOp',cid),'sgl-kernel-npu')()")
code=code.replace("'factory_called':True","'factory_called':True,'source_sha256':cand.source_sha256") if False else code
code=code.replace('CPU_FACTORY_CHECK.json','PARENT_LIVE_FACTORY_CHECK.json')
p=subprocess.run([sys.executable,'-S','-'],input=code,text=True,capture_output=True,timeout=30);assert p.returncode==0,p.stderr
route=json.loads(p.stdout);assert route['exact_keys']==9 and not route['torch_npu_imported']
receipts={};source_hashes={};native=[];checked=0;out=[];original_sources={}
specs={}
for f in ['convolution','sampling']:specs.update(yaml.safe_load((W/'TileOPs/src/tileops/manifest/spec'/f'{f}.yaml').read_text()))
for row in rs:
 k=[row['key'][x] for x in ['baseline','operator','case_id']];cp=Path(row['checker_receipt']);c=read(cp);gp=Path(row['enclosing_guard_receipt']);g=read(gp);v=c['verification']
 assert [c['library'],c['op'],c['case_id']]==k
 assert c['status']=='verified' and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
 assert v['checked_outputs']==v['total_outputs']==(1 if k[1]=='Conv2dFwdOp' else 2) and v['unchecked_reason'] is None
 checked+=v['checked_outputs'];assert c['input_seed']==int.from_bytes(hashlib.sha256((k[1]+'::'+k[2]).encode()).digest()[:4],'big')
 assert g['returncode']==0 and read(row['guard_receipt'])['returncode']==0
 assert '/batch04/' not in str(gp) and '/batch05/' not in str(gp)
 assert g['started']<=c['started_at']<=c['finished_at']<=g['finished']
 gm=read(cp.parent/'GUARD_RUNNING.json');assert gm['owner']=='baseline700-20261026/sgl_conv_sampling02' and gm['phase']=='running'
 for field in ['source_sha256','oracle_source_sha256']:
  for rel,h in c[field].items():
   pp=W/rel;assert sha(pp)==h,str(pp);source_hashes[str(pp)]=h
 assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py')
 for pp in [cp,gp,Path(row['guard_receipt']),cp.parent/'GUARD_RUNNING.json']:receipts[str(pp)]=sha(pp)
 workload=next(x for x in specs[k[1]]['workloads'] if k[2]==x['label'] or any(k[2]==x['label']+'-'+d['T'] for d in x.get('dtype_cases',[]) if 'T' in d))
 snap=c['case_snapshot'];obs=c['observed_calls'];assert obs['reference'][0]['returned'] and obs['candidate'][0]['returned']
 assert obs['reference'][0]['inputs']==obs['candidate'][0]['inputs']
 inputs=obs['candidate'][0]['inputs']['items']
 if k[1]=='Conv2dFwdOp':
  dtype=k[2].rsplit('-',1)[1]
  assert inputs[0]['shape']==[workload[t] for t in ['N','C_in','H','W']]
  assert inputs[1]['shape']==[workload[t] for t in ['C_out','C_in','kH','kW']]
  assert inputs[0]['dtype']==inputs[1]['dtype']==dtype
  assert (inputs[2]['kind']=='tensor')==('bias' in workload.get('some',[]))
  assert snap['parameters']['stride']==workload['stride'] and snap['parameters']['padding']==workload['padding']
  expected={'sgemmc_expand_1'}
  assert 'libsgl_kernel_npu.so' in (cp.parent/'maps.txt').read_text()
 else:
  b,n,z=[workload[t] for t in ['B','N','V']];assert inputs[0]['shape']==[b,n,z] and inputs[2]['shape']==[b,n+1,z]
  assert inputs[0]['dtype']==inputs[2]['dtype']=='float32'
  expected={'_chain_rejection_accept_kernel','_chain_rejection_block_sum_kernel','_chain_rejection_sample_kernel'}
 found={}
 for log in (cp.parent/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'):
  with log.open(errors='replace') as fh:
   for ln,line in enumerate(fh,1):
    if 'Launch' not in line:continue
    m=re.search(r'kernel_name=([^,]+)',line)
    if not m or m[1] not in expected or m[1] in found:continue
    dt=re.search(r'(2026-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',line)
    assert dt,line
    when=f'{dt[1]}T{dt[2]}.{dt[3]}{dt[4]}+00:00'
    if c['started_at']<=when<=c['finished_at']:found[m[1]]={'path':str(log),'line':ln,'text':line.strip(),'time':when}
  if set(found)==expected:break
 assert set(found)==expected,(k,found)
 native.append({'key':k,'launches':found})
 out.append({'key':k,'checker':{'path':str(cp),'sha256':sha(cp)},'guard':{'path':str(gp),'sha256':sha(gp),'exit_code':0},'verification':v,'inputs_preserved':True,'seed':c['input_seed'],'canonical':workload,'api':c['api'],'formal_registration':{'factory':row['factory'],'module':str(module),'module_sha256':sha(module)},'original_boundary':'Full original sgemmc_expand per independent spatial vector/fullK with bias supplied as original residual; or full original chain_speculative_sampling_triton with top-level uniform RNG and linear-chain metadata only.','source_sha256':c['source_sha256'],'oracle_source_sha256':c['oracle_source_sha256']})
for pp,h in source_hashes.items():
 prefix=str(W/'ascend_baselines/sgl-kernel-npu')+'/'
 if pp.startswith(prefix) and Path(pp).suffix in ['.py','.cpp','.h','.hpp']:
  rel=pp[len(prefix):];p=subprocess.run(['git','-C',prefix,'show','HEAD:'+rel],capture_output=True);assert p.returncode==0,rel
  assert hashlib.sha256(p.stdout).hexdigest()==h,rel;original_sources[rel]=h
assert checked==11 and len(original_sources)>10
put(D/'PARENT_SGL_CONV9_NATIVE.json',native)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();reviewpath=D/'PARENT_SGL_CONV9_REVIEW.json'
report={'at':now,'scope':9,'checked_outputs':11,'native_launch_evidence':str(D/'PARENT_SGL_CONV9_NATIVE.json'),'receipt_hashes':receipts,'critical_source_hashes':source_hashes,'original_git_HEAD_sources':original_sources,'live_registry_routes':str(P/'PARENT_LIVE_FACTORY_CHECK.json'),'adapter_byte_identical':True,'device_runs':0,'excluded_guards':['batch04 exit76','batch05 exit76'],'boundary':'7Conv full original channel reduction and bias residual per independent spatial vector, no splitK or kernel edits;2Chain full original public API including rejection/residual/categorical computations. RNG/linear-chain metadata only in adapter. Statistical validator plus distribution probe unchanged; large raw token max_abs difference is expected and not numeric closeness criterion.','waste':'Inherited unrelated custom OPP caused initial diagnostic failures. attempt03 started before environment fix; contaminated76 attempts retried with clean independent guard, no clean successful key rerun.','timings':read(O/'FINAL_CHECKS.json')}
put(reviewpath,report)
ledgerpath=D/'ACCEPTED.json'
with open(str(ledgerpath)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(ledgerpath);by={tuple(r['key']):r for r in ledger['rows']};before=len(by)
 for row in out:
  k=tuple(row['key'])
  if k not in by:row.update(accepted=True,at=now,parent_review=str(reviewpath));by[k]=row
 ledger.update(rows=list(by.values()),updated_at=now);put(ledgerpath,ledger)
report.update(accepted_added=len(by)-before,accepted_total=len(by));put(reviewpath,report);put(O/'PARENT_REVIEW.json',{'at':now,'scope':9,'accepted':9,'review':str(reviewpath)});put(P/'PARENT_REVIEW.json',{'at':now,'scope':9,'accepted':9,'review':str(reviewpath)})
s=read(D/'STATE.json');s.update(completed=len(by),parent_accepted_completed=len(by),last_status_accepted=len(by),last_status_at=now,last_acceptance_review=str(reviewpath));put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as a,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(a,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_conv9_'+now.replace(':','-')+'.json'));j=read(M);j['integration_700_20261026'].update(parent_accepted_completed=len(by),last_acceptance_review=str(reviewpath),last_status_at=now);put(M,j)
print(json.dumps({'accepted_added':len(by)-before,'accepted_total':len(by),'original_sources':len(original_sources),'receipt_hashes':len(receipts),'checked_outputs':checked}))
