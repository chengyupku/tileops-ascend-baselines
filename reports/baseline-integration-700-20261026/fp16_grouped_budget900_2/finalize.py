from pathlib import Path
import json,hashlib,time
from datetime import datetime
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rows=[]
for case in ['moe-dgrad-float16','moe-fwd-float16']:
 d=P/case;r=read(d/'result/representative.json');e=read(d/'result/EXECUTION.json');g=read(d/'GUARD_FINAL.json');n=read(d/'result/NATIVE_CALLS.json');cap=read(d/'result/CAPTURE.json');m=g['metadata']; snap=read(d/'result/guard_snapshot.json')
 entries=[x for x in n if x['event']=='enter'];returns=[x for x in n if x['event']=='return']
 assert r['status']=='verified' and r['inputs_preserved'] and e['frozen_unchanged'] and g['exit_code']==0
 assert len(entries)==len(returns)==32 and all(x['return_code']==0 for x in returns)
 assert not r['provider_source_changed'] and not r['oracle_source_changed']
 lines=[]
 for f in sorted((d/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
  for line in f.read_text(errors='replace').splitlines():
   if 'e84aaec0eccf7b8304da51ea5f5a2076' in line and ('Register Kernel' in line or 'StreamLaunchKernel' in line):lines.append({'path':str(f),'line':line})
 assert any('Register Kernel' in x['line'] for x in lines) and sum('StreamLaunchKernel' in x['line'] for x in lines)==16
 dump(d/'DEVICE_EVIDENCE.json',lines)
 for k,v in cap.items():
  if k.endswith('_output'):assert v['stored'] and sha(d/'result'/f'{k}.bin')==v['sha256'] and (d/'result'/f'{k}.bin').stat().st_size==v['bytes']
 delta=lambda a,b:(datetime.fromisoformat(a)-datetime.fromisoformat(b)).total_seconds()
 row={'case_id':case,'status':'verified_pending_parent_review','guard_rc':0,'verification':r['verification'],'inputs_preserved':True,'timings_seconds':e['timings'],'guard_wall_seconds':g['wall_seconds'],'queue_to_lock_seconds':delta(m['acquired_at'],m['requested_at']),'request_to_child_running_seconds':delta(snap['updated_at'],m['requested_at']),'native_size_calls':16,'native_launch_calls':16,'native_all_rc_zero':True,'e84_stream_launches':16,'frozen_unchanged':True,'output_full_bytes_each':33554432}
 dump(d/'RESULT.json',row);rows.append(row)
pins=read(P/'FROZEN.json')['sha256'];after={name:sha(Path(name)) for name in pins};changed={name:{'before':pins[name],'after':h} for name,h in after.items() if pins[name]!=h}
dump(P/'PINS_AFTER.json',{'sha256':after,'changed':changed});assert not changed
result={'status':'verified_pending_parent_review','scope_size':2,'verified':2,'accepted_added':0,'registry_published':False,'rows':rows,'remaining_issues':['Parent review and publication are pending.'],'notes':['First initial JSONL heartbeat sampled stale prior guard metadata before own reservation; all later samples identify actual child. HEARTBEAT.json added at second-case boundary following FEEDBACK.','Original mathematical harness, adapter, precision, seed, reference thread cap and input-preservation checks unchanged. Only guard deadline extended from300 to900 seconds.']}
dump(P/'RESULT.json',result)
dump(P/'DELIVERY.json',{'status':result['status'],'result':'RESULT.json','cpu_checks':'CPU_CHECKS.json','frozen':'FROZEN.json','pins_after':'PINS_AFTER.json','filehashes':'FILEHASHES.json','case_directories':[r['case_id'] for r in rows],'changed_files_scope':str(P),'shared_registry_changed_by_worker':False,'memory_changed':False,'npu_invocations':2,'reruns':0,'checks':['Existing CPU contract mocks passed','3465 pins checked before and after; unchanged during execution','Both original fullcheckers verified with inputs_preserved','All32 V5 size and32 launch returns zero across two cases','Each case e84 registration and16 StreamLaunchKernel receipts','Full reference and candidate output binaries hash-verified'],'remaining_issues':result['remaining_issues']})
dump(P/'HEARTBEAT.json',{'at':time.time(),'phase':'complete','child_pid':None,'last_progress':'Both cases verified; delivery artifacts generated','status':result['status']})
files={str(f.relative_to(P)):sha(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='FILEHASHES.json'}
dump(P/'FILEHASHES.json',{'sha256':files,'excludes':['FILEHASHES.json (self)'],'created_at':time.time()})
print(json.dumps(result,indent=2));print('files',len(files))
