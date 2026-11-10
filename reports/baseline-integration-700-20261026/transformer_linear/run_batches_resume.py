from pathlib import Path
import json,subprocess,sys,time,datetime
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'chunk_package'));from resource_gate import require_capacity
ids=json.loads((W/'CHUNK_CASES.json').read_text());est=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text());ids.sort(key=lambda i:est[i]['declared_tensor_bytes'])
blocked=json.loads((W/'BLOCKERS.json').read_text()) if (W/'BLOCKERS.json').exists() else {}
ready=[]
for i in ids:
 if (W/i/'GUARD_RECEIPT.json').exists():continue
 try:require_capacity(W/'chunk_package',W/i);ready.append(i)
 except Exception as e:blocked['GDNFwdOp/'+i]={'reason':str(e),'api':'aclnnChunkGatedDeltaRule','evidence':str((W/i/'fresh_resource_admission.json').relative_to(W))}
(W/'BLOCKERS.json').write_text(json.dumps(blocked,indent=2));subprocess.run([sys.executable,str(W/'update_result.py')],check=True)
for b in range(0,len(ready),3):
 group=ready[b:b+3];now=datetime.datetime.now(datetime.timezone.utc).isoformat();(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='batch_guard',current_keys=[['ops-transformer','GDNFwdOp',i] for i in group],last_progress_at=now,blocked_reason=None),indent=2))
 if (W/'FEEDBACK.txt').exists():print((W/'FEEDBACK.txt').read_text(),flush=True)
 d=W/('batch_resume_'+str(b//3));d.mkdir(exist_ok=True)
 child=(W/group[0]/'child.sh').read_text();child=child[:child.index('exec /opt/tileops-ci')]+f'exec /opt/tileops-ci/runtime/bin/python -S -B "{W}/batch_entry.py" '+ ' '.join(group)+'\n';(d/'child.sh').write_text(child)
 argv=['npu-guard','run','--owner','baseline700-20261026/transformer_linear','--wait','3600','--timeout','300','--','bash',str(d/'child.sh')];t=time.time();print('BATCH',group,flush=True)
 with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
 end=time.time();receipt=dict(argv=argv,start=now,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,elapsed_seconds=end-t)
 (d/'GUARD_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
 for i in group:
  if (W/i/'DEVICE_STARTED.json').exists():
   ts=json.loads((W/i/'DEVICE_STARTED.json').read_text())['timestamp'];te=json.loads((W/i/'DEVICE_FINISHED.json').read_text())['timestamp'] if (W/i/'DEVICE_FINISHED.json').exists() else None
   rec=dict(receipt,queue_seconds=ts-t if i==group[0] else None,device_seconds=te-ts if te else None,batch_receipt=str(d.relative_to(W)))
   (W/i/'GUARD_RECEIPT.json').write_text(json.dumps(rec,indent=2))
 subprocess.run([sys.executable,str(W/'register.py')],check=True);subprocess.run([sys.executable,str(W/'update_result.py')],check=True)
 if p.returncode:
  print('SHARED FAILURE STOP',p.returncode,flush=True);break
