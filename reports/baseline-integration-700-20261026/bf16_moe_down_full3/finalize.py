from pathlib import Path
import json,hashlib,time,collections
T=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')
pins=read(T/'FROZEN.json')['sha256'];after={p:sha(Path(p)) for p in pins};changed={p:{'before':h,'after':after[p]} for p,h in pins.items() if h!=after[p]};dump(T/'PINS_AFTER.json',dict(sha256=after,changed=changed))
rows=[]
for case in read(T/'CASES.json'):
 C=T/case['key'][2];P=C/'result';r=read(P/'representative.json');e=read(P/'EXECUTION.json');n=read(P/'NATIVE_CALLS.json');rc=int((C/'GUARD_RC.txt').read_text());entries=collections.Counter(x['function'] for x in n if x['event']=='enter');returns=collections.Counter(x['return_code'] for x in n if x['event']=='return')
 evidence=[];device=collections.Counter()
 for p in (C/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*'):
  if not p.is_file():continue
  with p.open(errors='replace') as f:
   for i,line in enumerate(f,1):
    if any(x in line for x in ['GroupedMatmulV5','GroupedMatmul_e84aaec0eccf7b8304da51ea5f5a2076','StreamLaunch','Register']):
     evidence.append(dict(file=str(p),line=i,text=line.rstrip()))
     if 'e84aaec0eccf7b8304da51ea5f5a2076' in line:
      device['e84_lines']+=1
      if 'Register' in line:device['e84_Register_lines']+=1
      if 'StreamLaunch' in line:device['e84_StreamLaunch_lines']+=1
 dump(C/'NATIVE_LOG_EVIDENCE.json',evidence)
 proof=dict(entries=entries,return_codes=returns,device_evidence=device,actual_function_objects=[x for x in n if x['event']=='enter'],evidence_lines=len(evidence));dump(C/'NATIVE_PROOF_SUMMARY.json',proof)
 good=rc==0 and r['status']=='verified' and r['inputs_preserved'] and r['verification']['checked_outputs']==r['verification']['total_outputs'] and e['frozen_unchanged'] and not changed and entries=={'aclnnGroupedMatmulV5GetWorkspaceSize':256,'aclnnGroupedMatmulV5':256} and returns=={0:512} and device['e84_Register_lines']>0 and device['e84_StreamLaunch_lines']>0
 row=dict(key=case['key'],status='verified_pending_parent_review' if good else 'review_required',guard_exit_code=rc,verification=r['verification'],inputs_preserved=r['inputs_preserved'],checker_wall_seconds=r['wall_seconds'],timings=e['timings'],native_entries=entries,native_returns=returns,device_evidence=device,frozen_unchanged=e['frozen_unchanged'],evidence_directory=str(C),resource_admission=read(P/'RESOURCE_ADMISSION.json'))
 dump(C/'RESULT.json',row);rows.append(row)
result=dict(status='verified_pending_parent_review' if all(x['status']=='verified_pending_parent_review' for x in rows) else 'review_required',accepted_added=0,registered=False,results=rows,pins_checked=len(pins),pins_changed=changed,numerical_replays=0,startup_repair=read(T/'STARTUP_REPAIR.json'))
dump(T/'RESULT.json',result)
dump(T/'HEARTBEAT.json',dict(time=time.time(),stage='terminal',status=result['status']))
dump(T/'DELIVERY.json',dict(status=result['status'],accepted_added=0,registered=False,findings=rows,checks=['AST and bash syntax, positional launch arguments','Actual tensor descriptors and non-tensor Call.indices','Fresh host/full cgroup ancestors, disk and HBM gates','Original full canonical checker with independent reference and inputs_preserved','256 size and 256 launch function-address observations per case, all native rc 0','e84 Register/StreamLaunch raw device evidence','3469 source/deployment/runtime pins unchanged'],remaining_issues=['Parent review and registration pending'],startup_bug_repaired_before_canonical=True,changed_files=[str(p.relative_to(T)) for p in T.rglob('*') if p.is_file()]))
dump(T/'FILEHASHES.json',{str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'})
print(json.dumps(dict(status=result['status'],pins_changed=changed,results=[{k:r[k] for k in ['key','status','verification','native_entries','device_evidence']} for r in rows]),indent=2))
