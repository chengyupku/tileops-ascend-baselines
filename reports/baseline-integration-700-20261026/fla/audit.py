import pathlib,json,hashlib,collections,datetime
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
rows=json.loads((p/'RESULT.json').read_text())['rows'];versions={hashlib.sha256(f.read_bytes()).hexdigest():str(f) for f in [p/'integration700_fla_native.first_registered.py',p/'source_snapshot_v2.py',p/'frozen_adapter.py']};audits=[]
for r in rows:
 if r['status']!='completed':continue
 d=json.loads(pathlib.Path(r['checker_receipt']).read_text());g=json.loads(pathlib.Path(r['guard_receipt']).read_text())
 assert d['status']=='verified' and not d['provider_source_changed'] and not d['oracle_source_changed'],r
 assert d['verification']['checked_outputs']==d['verification']['total_outputs']>0
 assert g['returncode'] not in (75,76,78,124),r
 h=next((v for k,v in d.get('source_sha256',{}).items() if k.endswith('/integration700_fla_native.py')),None)
 if h is None:
  assert r['operator']=='GemmFwdOp'
  entry='TileOPs/benchmarks/ascend_all_pairs/fla/cube_matmul.py';h=d['source_sha256'][entry];assert hashlib.sha256((root/entry).read_bytes()).hexdigest()==h
  snapshot=str(p/'source_snapshot_cube_matmul.py')
 else:
  assert h in versions,(r['case_id'],h)
  snapshot=versions[h]
 audits.append(dict(operator=r['operator'],case_id=r['case_id'],checker=r['checker_receipt'],guard=r['guard_receipt'],guard_returncode=g['returncode'],all_outputs_checked=True,source_snapshot=snapshot,source_sha256=h))
(p/'FINAL_AUDIT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=len(audits),rows=audits,scope_count=44,status_counts=dict(collections.Counter(r['status'] for r in rows))),indent=2));print('Audited completed',len(audits))
