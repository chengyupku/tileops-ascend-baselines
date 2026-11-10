from pathlib import Path
import json,hashlib,datetime,collections
W=Path(__file__).resolve().parent
scope=json.loads((W/'SCOPE.json').read_text());registered=json.loads((W/'REGISTERED.json').read_text()) if (W/'REGISTERED.json').exists() else []
blocked=json.loads((W/'BLOCKERS.json').read_text()) if (W/'BLOCKERS.json').exists() else {}
def receipt(f):return {'path':str(f.relative_to(W)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} if f.exists() else None
rows=[]
for x in scope:
 op=x['operator'];i=x['case_id'];key=op+'/'+i;d=W/i;f=d/'checker.json';g=d/'GUARD_RECEIPT.json'
 if op in ('GDNFwdOp','CausalConv1dUpdateFwdOp','SSDChunkStateFwdOp') and f.exists() and g.exists():
  c=json.loads(f.read_text());guard=json.loads(g.read_text());v=c.get('verification',{});ok=c.get('status')=='verified' and guard['returncode']==0 and v.get('checked_outputs')==v.get('total_outputs')==2
  row=dict(status=('completed' if i in registered else 'verified_unregistered') if ok else 'failed',reason=c.get('error') or ('Unrelated concurrent provider registration changed shared provider snapshot; full numerical outputs passed but not accepted as completed.' if c.get('status')=='source_changed' else None) or (None if ok else f"guard exit {guard['returncode']}"),api=c.get('api','aclnnChunkGatedDeltaRule'),guard_receipt=receipt(g),checker_receipt=receipt(f),checked_outputs=v.get('checked_outputs',0),total_outputs=1 if op=='SSDChunkStateFwdOp' else 2,registered=ok and i in registered,factory='benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn.build' if ok and i in registered else None,changed_files=([('chunk_package' if op=='GDNFwdOp' else 'causal_package' if op=='CausalConv1dUpdateFwdOp' else 'ssd_package')+'/'+n for n in ['binding.py','entry.py']]+[i+'/child.sh']+(['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_linear_gdn.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py'] if ok and i in registered else [])),timings={'queue_seconds':guard.get('queue_seconds'),'device_seconds':guard.get('device_seconds'),'compile_seconds':None,'total_seconds':guard.get('elapsed_seconds'),'checker_seconds':c.get('wall_seconds')})
 elif key in blocked:
  b=blocked[key];row=dict(status='blocked',reason=b['reason'],api=b.get('api'),guard_receipt=b.get('guard_receipt'),checker_receipt=b.get('checker_receipt'),checked_outputs=0,total_outputs=1 if op=='SSDChunkStateFwdOp' else 2,registered=False,factory=None,changed_files=[],timings={'queue_seconds':None,'device_seconds':None,'compile_seconds':None},evidence=b.get('evidence'))
 else:continue
 rows.append(dict(baseline=x['baseline'],operator=op,case_id=i,**row))
v=dict(host='server2',scope_count=len(scope),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=dict(collections.Counter(x['status'] for x in rows)),rows=rows,remaining_unresolved=len(scope)-len(rows));t=W/'RESULT.tmp';t.write_text(json.dumps(v,indent=2));t.replace(W/'RESULT.json');print(v['counts'],'unresolved',v['remaining_unresolved'],flush=True)

if (W/"RECONFIGURE_BATCH_SIZE").exists():
 print("End current batch cleanly to amortize measured queue overhead",flush=True)
 raise SystemExit(77)
