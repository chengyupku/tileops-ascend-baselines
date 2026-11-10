import pathlib,json,datetime
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'SCOPE.json').read_text())
for r in rows:
 r['prior_attempt']={k:r.get(k) for k in ['status','reason','guard_receipt','checker_receipt','timings']}
 r.update(status='blocked',registered=False,checked_outputs=0,guard_receipt=None,checker_receipt=None,timings=dict(queue_seconds=None,device_seconds=None,compile_seconds=None))
 f=P/'runs'/('tilelang_'+r['operator']+'_'+r['case_id'])/'checker.json'
 if r['baseline']=='tilelang' and f.exists():
  d=json.loads(f.read_text());v=d.get('verification',{})
  r.update(status='verified_unregistered' if d['status']=='verified' else 'failed',reason='CANN 9.2.0-beta.2 original SDK environment resolves 9.1 compilation failure; full canonical checker passed; parent registration required.' if d['status']=='verified' else d.get('error','checker failed'),api=d.get('api'),checker_receipt=str(f),guard_receipt=str(f.parent/'GUARD_RUNNING.json'),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs'),factory='integration700_tla_sdk_recovery36.factory',changed_files=['run_tilelang.py'])
  meta=json.loads((f.parent/'GUARD_RUNNING.json').read_text())
  t=lambda s:datetime.datetime.fromisoformat(s).timestamp()
  r['timings']=dict(queue_seconds=t(meta['acquired_at'])-t(meta['requested_at']),device_seconds=d.get('wall_seconds'),compile_seconds=None)
 elif r['operator']=='TopKSelectFwdOp':
  r['reason']='Actual native compile failure is PTO Tci.hpp:30 static_assert: TCI only supports uint16/32 or int16/32, generated TopK uses float. SDK symbol upgrade does not alter this vendor type requirement; kernel/compiler modification forbidden.'
 elif r['operator']=='IsnanFwdOp':
  r['reason']='Canonical inputs and reference completed. Adapter output unpack shifts=torch.arange(8,dtype=torch.uint8,device=y.device) fails unsupported aclnnArange dtype; independent subsequent synchronize reports 507015. Fixing shift construction alone does not establish backend repair; no unchanged backend rerun.'
 r['evidence_current']=['sdk_inventory.txt','compile_recovered.txt','prior_evidence']
tmp=P/'RESULT.tmp';tmp.write_text(json.dumps(dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows),indent=2));tmp.replace(P/'RESULT.json')
print({s:sum(r['status']==s for r in rows) for s in ['verified_unregistered','failed','blocked']})
