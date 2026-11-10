from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;scope=json.loads((P/'SCOPE.json').read_text());rows=[]
receipts=list((P/'runs').glob('*/output/checker.json'))+list((P/'batches').glob('*/*/checker.json'))
for s in scope:
 hits=[]
 for p in receipts:
  d=json.loads(p.read_text())
  if [d.get('library'),d.get('op'),d.get('case_id')]==s['key']:
   guard=p.parents[1]/'GUARD_RESULT.json'
   gd=json.loads(guard.read_text()) if guard.exists() else {'returncode':None}
   hits.append((p,d,guard,gd))
 successes=[h for h in hits if h[1].get('status')=='verified' and h[3]['returncode']==0 and h[1].get('verification',{}).get('checked_outputs')==h[1].get('verification',{}).get('total_outputs')==1]
 chosen=successes[-1] if successes else hits[-1] if hits else None
 row={'key':s['key'],'status':'verified' if successes else 'blocked','api':'ops-transformer::FlashAttentionScore via CANN graph JIT','changed_files':['ascend_graph_call.py','run_case.py'],'total_outputs':1,'checked_outputs':1 if successes else 0,'attempt_receipts':[str(h[0]) for h in hits],'already_completed_skipped':False}
 if chosen:
  p,d,g,gd=chosen;row.update(checker_receipt=str(p),guard_receipt=str(g),guard_exit_code=gd['returncode'],inputs_preserved=d.get('inputs_preserved'),reason=None if successes else d.get('error','Guard did not exit 0'))
 else:row['reason']='Stopped after common route failure; see prior attempted-case receipt.'
 rows.append(row)
result={'host':'server2','owner':'selected-integration-20261026/transformer_attention','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'verified':sum(r['status']=='verified' for r in rows),'scope_count':len(rows),'guard_timeout':300,'checker':'/opt/tileops-all-pairs/workspace/TileOPs/scripts/ascend_pair_check.py','checker_unmodified':True,'vendor_kernel_changes':False,'benchmark_requested':False,'final_state_or_cache_outputs':'not part of GQADenseFwdOp contract; full o checked','source_evidence':str(P/'SOURCE_EVIDENCE.json')}
(P/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
changed=['ascend_graph_call.py','ascend_transformer_graph.py','ascend_transform_baselines.py','run_case.py','run_batch.py','launch.py','launch_remaining.py','provider.patch']
delivery={'status':'complete' if result['verified']==len(rows) else 'partial_blocked','host':'server2','callable_entry':'run_case.py installs ops_transformer.PROVIDERS[GQADenseFwdOp] = transformer_graph_gqa; original complete FlashAttentionScore API via isolated GraphCall vendor path','adapter':str(P/'ascend_transformer_graph.py')+':transformer_graph_gqa','patch':str(P/'provider.patch'),'registration':str(P/'run_case.py'),'run_method':'python3 '+str(P/'launch.py')+' CASE_ID NEW_ATTEMPT_TAG (internally npu-guard run --owner selected-integration-20261026/transformer_attention --wait 3600 --timeout 300); do not rerun verified keys','existing_guarded_commands':[str(p) for p in sorted(P.glob('runs/*/COMMAND.json'))]+[str(p) for p in sorted(P.glob('batches/*/COMMAND.json'))],'changed_files':changed,'merge_note':'Private overlay only; shared provider/registry/vendor/environment unchanged. Vendor path is server2 installed original package in VENDOR.json; retain explicit isolated deployment when merging. No memory updates.','unresolved':[r for r in rows if r['status']!='verified'],'result':str(P/'RESULT.json'),'source_evidence':str(P/'SOURCE_EVIDENCE.json'),'checked_outputs':sum(r['checked_outputs'] for r in rows),'total_outputs':len(rows)}
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2)+'\n')
print(json.dumps({'verified':result['verified'],'scope_count':len(rows)}))
