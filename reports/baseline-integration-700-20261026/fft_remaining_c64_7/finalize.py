"""CPU-only evidence indexing; no runtime imports or device execution."""
import json,hashlib,re,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace')
def read(p): return json.loads(p.read_text()) if p.exists() else {}
def write(p,x):p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(P/'RESULT.json');assert not r['in_progress']
freeze=read(P/'FROZEN.json');assert all(sha(P/n)==v for n,v in freeze.items())
sources=read(P/'SOURCE_BEFORE.json');compare=[{'path':n,'before':v,'after':sha(W/n),'unchanged':sha(W/n)==v} for n,v in sources.items()]
write(P/'SOURCE_COMPARE_AFTER.json',compare)
for row in r['rows']:
 d=P/row['case_id']
 if not d.exists():continue
 c=read(d/'checker.json');g=read(d/'GUARD_RESULT.json');payload=read(d/'PAYLOADS.json');m=read(d/'GUARD_METADATA.json')
 events=[];logs=[]
 pattern=re.compile(r'STFT|Stft|libcpu_kernels|RunCpuKernel|Load so|OpenSo|Register.*[Kk]ernel')
 for f in sorted((d/'ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
  logs.append({'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size})
  for i,line in enumerate(f.read_text(errors='replace').splitlines(),1):
   if pattern.search(line):events.append({'file':str(f),'line':i,'text':line})
 (d/'native.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
 write(d/'NATIVE_EVIDENCE_INDEX.json',logs)
 retained=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(d.glob('*.pt'))]
 write(d/'PAYLOAD_INDEX.json',retained)
 row.update(verification=c.get('verification'),seed=c.get('input_seed'),inputs_preserved=c.get('inputs_preserved'),payloads=payload,retained_payload_index=str(d/'PAYLOAD_INDEX.json'),native_index=str(d/'NATIVE_EVIDENCE_INDEX.json'),native_events=str(d/'native.jsonl'),source_sha256=c.get('source_sha256'),timings={'guard_total_wall_s':g.get('elapsed_s'),'checker_wall_s':c.get('wall_seconds'),'compile_s':None,'cpu_preparation_s':None,'queue_s':None},cleanup={'final_sync_and_candidate_cleanup':'passed' if c.get('status')=='verified' else 'see checker','status_after':str(d/'STATUS_AFTER.json')})
 if m.get('requested_at') and m.get('updated_at'):row['timings']['guard_request_to_running_s']=(datetime.datetime.fromisoformat(m['updated_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds()
 if row['status']=='verified_unregistered':
  assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
  assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
  assert payload and all(v['input_preserved'] for v in payload)
 write(d/'RESULT.json',row)
write(P/'RESULT.json',r)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
delivery={'task':P.name,'worker':'17b45d1f18b64176','finished_at':now,'scope_count':7,'npu_attempts':sum((P/x['case_id']/'COMMAND.json').exists() for x in r['rows']),'verified_unregistered':sum(x['status']=='verified_unregistered' for x in r['rows']),'new_parent_acceptances':0,'registered':False,'result':str(P/'RESULT.json'),'findings':['Independent canonical full checks, original seeds/shapes/dtypes/reference/verifier retained.','Private binding preserves batch axis, supplies exact same-dtype complex unity window, calls original complete aclStft and only reshapes output.','Actual runtime attribution is system CANN libcpu_kernels.so::RunCpuKernel STFT, never custom libmath_aicpu_kernels.so. Native raw host/device logs and indexed lines retained per case.'],'remaining_issues':['Parent acceptance and CPU registration remain pending.','Loaded device libcpu_kernels.so is not host-readable based on prior diagnosis; independent device-binary hash unavailable, no rerun or SDK scan performed.','Correctness only; no performance measurement.'],'not_done':[x['case_id'] for x in r['rows'] if x['status']=='blocked'],'shared_source_modified':False,'registry_modified':False,'memory_modified':False,'source_hashes_unchanged':all(x['unchanged'] for x in compare),'changed_files':[str(f.relative_to(P)) for f in sorted(P.rglob('*')) if f.is_file() and f.name not in ['SCOPE.json','SPAWN.json','TASK.txt']]}
write(P/'DELIVERY.json',delivery)
write(P/'HEARTBEAT.json',{'time':now,'phase':'finished','verified_unregistered':delivery['verified_unregistered'],'new_parent_acceptances':0})
write(P/'EVIDENCE_INDEX.json',[{'path':str(f.relative_to(P)),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(P.rglob('*')) if f.is_file() and f.name!='EVIDENCE_INDEX.json'])
print(json.dumps(delivery,indent=2))
