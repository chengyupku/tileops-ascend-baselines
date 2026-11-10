from pathlib import Path
import json,hashlib,time
from datetime import datetime
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
rows=[]
for case in ['moe-fwd-bfloat16','square-4k-bfloat16']:
 d=P/case
 if not (d/'GUARD_FINAL.json').exists():
  rows.append({'case_id':case,'status':'not_run','reason':'Prior case did not fully verify; conditional execution gate retained.'});continue
 g=read(d/'GUARD_FINAL.json');rp=d/'result/representative.json'
 r=read(rp) if rp.exists() else {};ep=d/'result/EXECUTION.json';e=read(ep) if ep.exists() else {}
 np=d/'result/NATIVE_CALLS.json';n=read(np) if np.exists() else []
 entries=[x for x in n if x['event']=='enter'];returns=[x for x in n if x['event']=='return']
 lines=[]
 for f in sorted((d/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*.log')):
  for line in f.read_text(errors='replace').splitlines():
   if 'e84aaec0eccf7b8304da51ea5f5a2076' in line and ('Register Kernel' in line or 'StreamLaunchKernel' in line):lines.append({'path':str(f),'line':line})
 dump(d/'DEVICE_EVIDENCE.json',lines)
 verified=r.get('status')=='verified' and r.get('inputs_preserved') and e.get('frozen_unchanged') and g['exit_code']==0
 if verified:
  assert entries and len(entries)==len(returns) and all(x['return_code']==0 for x in returns)
  assert not r['provider_source_changed'] and not r['oracle_source_changed']
  assert any('Register Kernel' in x['line'] for x in lines) and any('StreamLaunchKernel' in x['line'] for x in lines)
  for k,v in read(d/'result/CAPTURE.json').items():
   if k.endswith('_output'):assert v['stored'] and sha(d/'result'/f'{k}.bin')==v['sha256'] and (d/'result'/f'{k}.bin').stat().st_size==v['bytes']
 m=g['metadata'];snap=read(d/'result/guard_snapshot.json') if (d/'result/guard_snapshot.json').exists() else {}
 delta=lambda a,b:(datetime.fromisoformat(a)-datetime.fromisoformat(b)).total_seconds() if a and b else None
 status='verified_pending_parent_review' if verified else ('reference_timeout' if g['exit_code']==124 and not entries else r.get('status','guard_or_execution_failed'))
 row={'case_id':case,'status':status,'guard_rc':g['exit_code'],'verification':r.get('verification'),'inputs_preserved':r.get('inputs_preserved'),'timings_seconds':e.get('timings'),'guard_wall_seconds':g['wall_seconds'],'queue_to_lock_seconds':delta(m.get('acquired_at'),m.get('requested_at')),'request_to_child_running_seconds':delta(snap.get('updated_at'),m.get('requested_at')),'native_calls':len(entries),'native_all_rc_zero':bool(returns) and all(x['return_code']==0 for x in returns),'e84_stream_launches':sum('StreamLaunchKernel' in x['line'] for x in lines),'frozen_unchanged':e.get('frozen_unchanged')}
 dump(d/'RESULT.json',row);rows.append(row)
pins=read(P/'FROZEN.json')['sha256'];after={name:sha(Path(name)) for name in pins};changed={name:{'before':pins[name],'after':h} for name,h in after.items() if pins[name]!=h}
dump(P/'PINS_AFTER.json',{'sha256':after,'changed':changed})
verified=sum(r['status']=='verified_pending_parent_review' for r in rows)
status='verified_pending_parent_review' if verified==2 and not changed else 'stopped_pending_parent_review'
result={'status':status,'scope_size':2,'verified':verified,'accepted_added':0,'registry_published':False,'rows':rows,'pins_changed':changed,'remaining_issues':['Parent review and publication pending.'] if verified==2 else ['See per-case status; no oracle changes or reruns.'],'notes':['Original BF16 adapter body unchanged; exact factory entry injected original dense_fp32.','Original canonical seed, parameters, input-preservation checks, precision and four-thread reference retained. Only guard budgets extended.']}
dump(P/'RESULT.json',result)
dump(P/'DELIVERY.json',{'status':status,'result':'RESULT.json','cpu_checks':'CPU_CHECKS.json','frozen':'FROZEN.json','pins_after':'PINS_AFTER.json','filehashes':'FILEHASHES.json','case_directories':[r['case_id'] for r in rows],'changed_files_scope':str(P),'shared_registry_changed_by_worker':False,'memory_changed':False,'npu_invocations':sum('guard_rc' in r for r in rows),'reruns':0,'remaining_issues':result['remaining_issues']})
dump(P/'HEARTBEAT.json',{'at':time.time(),'stage':'complete','phase':'complete','child_pid':None,'status':status})
files={str(f.relative_to(P)):sha(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='FILEHASHES.json'}
dump(P/'FILEHASHES.json',{'sha256':files,'excludes':['FILEHASHES.json (self)'],'created_at':time.time()})
print(json.dumps(result,indent=2));print('files',len(files))
