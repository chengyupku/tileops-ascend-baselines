from pathlib import Path
import json,hashlib,datetime,difflib,ast
p=Path(__file__).parent
r=json.loads((p/'RESULT.json').read_text());out=[]
for row in r['rows']:
 if row['status']!='completed':continue
 m=row['factory'].split('.')[-2]+'.py';c=json.loads((p/row['checker_receipt']).read_text());tested=[v for k,v in c['source_sha256'].items() if k.endswith('/'+m)];assert len(tested)==1
 live=hashlib.sha256((p/m).read_bytes()).hexdigest();same=tested[0]==live
 proof=None if same else ('YARN_MIGRATION_PROOF.json' if row['operator']=='YaRNFwdOp' else 'SOURCE_MIGRATION_PROOF.json');assert same or row['operator'] in ['YaRNFwdOp','FusedTopKFwdOp'];out.append(dict(key=[row['baseline'],row['operator'],row['case_id']],module=m,tested_sha256=tested[0],registered_sha256=live,identical=same,migration_proof=proof))
(p/'ALL_FACTORY_HASHES.json').write_text(json.dumps(out,indent=2))
bins=[]
for f in sorted((p/'gemm/package').glob('*/GroupedMatmul*')):
 if f.suffix in ['.o','.json']:bins.append(dict(path=str(f.relative_to(p)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
(p/'ORIGINAL_DEVICE_BINARIES.json').write_text(json.dumps(bins,indent=2))
seen=set();rows=[]
for f in p.rglob('guard_snapshot.json'):
 s=json.loads(f.read_text());ticket=s.get('queue_ticket');
 if not ticket or ticket in seen or not s.get('acquired_at'):continue
 seen.add(ticket);rows.append(dict(snapshot=str(f.relative_to(p)),ticket=ticket,queue_s=(datetime.datetime.fromisoformat(s['acquired_at'])-datetime.datetime.fromisoformat(s['requested_at'])).total_seconds(),requested_at=s['requested_at'],acquired_at=s['acquired_at'],hold_s=None))
(p/'QUEUE_ACCOUNTING.json').write_text(json.dumps(dict(measured_unique_batches=len(rows),measured_queue_s=sum(x['queue_s'] for x in rows),rows=rows,limitation='Snapshot-covered batches only; exact held/device time not recorded and remains null. Batch/checker/compile wall times are separate. No cross-host clock subtraction.'),indent=2))
for f in p.glob('provider_init*.patch'):
 before=f.with_name(f.stem+'.before.py')
 if not before.exists():continue
 old=before.read_text();new=old+f.read_text();f.with_suffix('.unified.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py',tofile='b/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py')))
for f in p.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
print('audited',len(out),'factories;',len(rows),'unique guard snapshots;',sum(x['queue_s'] for x in rows),'queue seconds')
