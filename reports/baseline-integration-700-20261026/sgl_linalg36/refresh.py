import pathlib,json,datetime,hashlib,collections
p=pathlib.Path(__file__).resolve().parent;now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[]
for r in json.loads((p/'SCOPE.json').read_text()):
 op,cid=r['operator'],r['case_id'];found=[]
 for f in (p/'runs').glob('*/checker.json'):
  d=json.loads(f.read_text())
  if (d['op'],d['case_id'])!=(op,cid):continue
  g=f.parent/'GUARD_RESULT.json'
  if not g.exists():g=p/f.parent.name.split('--')[0]/'GUARD_RESULT.json'
  guard=json.loads(g.read_text()) if g.exists() else None
  found.append((d.get('status')=='verified' and guard is not None and guard['exit_code'] not in [75,76,78,124],d.get('finished_at',''),f,d,g,guard))
 found.sort(key=lambda x:x[:2]);chosen=found[-1] if found else None
 row={k:r[k] for k in ['baseline','operator','case_id']}
 row.update(status='blocked',reason='Awaiting expansion after verified representative; no numerical result for this key yet',api='torch.ops.npu.sgemmv_shrink',guard_receipt=None,checker_receipt=None,checked_outputs=0,total_outputs=1,registered=False,factory='integration700_sgl_linalg36.build',changed_files=['integration700_sgl_linalg36.py','run_case.py','baseline-run'],timings=dict(queue_s=None,device_s=None,compile_s=None))
 if chosen:
  valid,_,f,d,g,guard=chosen; v=d.get('verification',{})
  row.update(status='verified_unregistered' if valid else 'failed' if guard else 'blocked',reason='Complete canonical checker pass; task prohibits shared provider writes; registration patch for parent' if valid else d.get('error') or 'Awaiting clean completed guard receipt',api=d.get('api'),checker_receipt=str(f.relative_to(p)),guard_receipt=str(g.relative_to(p)) if g.exists() else None,checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',1),adapter_sha256=d.get('source_sha256',{}).get('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_linalg36/integration700_sgl_linalg36.py'),timings=dict(queue_s=guard.get('queue_s') if guard else None,device_s=None,compile_s=None,checker_wall_s=d.get('wall_seconds'),shared_guard_total_s=guard.get('total_s') if guard else None))
 rows.append(row)
t=p/'RESULT.json.tmp';t.write_text(json.dumps(dict(at=now,counts=dict(collections.Counter(x['status'] for x in rows)),rows=rows),indent=2));t.replace(p/'RESULT.json');print(collections.Counter(x['status'] for x in rows))
(p/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='batch_validation',current_keys=[(x['operator'],x['case_id']) for x in rows if x['status']!='verified_unregistered'],last_progress_at=now,blocked_reason=None,verified_unregistered=sum(x['status']=='verified_unregistered' for x in rows)),indent=2))
