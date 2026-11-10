import json,pathlib,datetime,collections
P=pathlib.Path(__file__).resolve().parent;rows=[]
for r in json.loads((P/'SCOPE.json').read_text()):
 op,cid=r['operator'],r['case_id']; receipts=sorted((P/'runs').glob('*--'+op+'--'+cid+'/checker.json')); row={k:r[k] for k in ['baseline','operator','case_id']};row.update(key=[r['baseline'],op,cid],status='pending',reason='Not run yet',checker_receipt=None,guard_receipt=None,registered=False,api='sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_native',timings={'queue_seconds':None,'device_seconds':None,'checker_seconds':None,'compile_seconds':None})
 if receipts:
  f=receipts[-1];c=json.loads(f.read_text());label=f.parent.name.split('--')[0];g=P/label/'GUARD_RESULT.json';guard=json.loads(g.read_text()) if g.exists() else {};v=c.get('verification',{})
  good=c.get('status')=='verified' and guard.get('exit_code')==0 and v.get('checked_outputs')==v.get('total_outputs')==2
  row.update(status='verified_unregistered' if good else 'blocked',reason='Original checker verifies both output and state; clean guard; pending parent registration' if good else c.get('error',c.get('reason','incomplete receipt')),checker_receipt=str(f.relative_to(P)),guard_receipt=str(g.relative_to(P)),api=c.get('api',row['api']),verification=v,timings={'queue_seconds':guard.get('queue_s'),'device_seconds':guard.get('device_s'),'checker_seconds':c.get('wall_seconds'),'compile_seconds':None})
 if cid=='decode-gate-fused-bfloat16':row.update(status='unsupported',reason='Original complete native/chunk APIs do not implement fused gate/beta semantics; no replacement math permitted.')
 rows.append(row)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/'RESULT.json').write_text(json.dumps({'utc':now,'rows':rows},indent=2));counts=dict(collections.Counter(r['status'] for r in rows));(P/'HEARTBEAT.json').write_text(json.dumps({'utc':now,'counts':counts},indent=2));print(counts)
