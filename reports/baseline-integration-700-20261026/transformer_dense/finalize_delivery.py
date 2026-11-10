from pathlib import Path
import json,hashlib,datetime,collections,sys
p=Path(__file__).parent;now=datetime.datetime.now(datetime.timezone.utc);r=json.loads((p/'RESULT.json').read_text());scope=json.loads((p/'SCOPE.json').read_text());keys={(x['baseline'],x['operator'],x['case_id']) for x in scope};assert len(keys)==134;assert {(x['baseline'],x['operator'],x['case_id']) for x in r['rows']}==keys
registered=set()
for f in p.glob('REGISTRATION*.json'):
 d=json.loads(f.read_text())
 if d.get('registered') and d.get('key'):registered.add(tuple(d['key']))
 for entry in d.get('rows',[]):
  if entry.get('registered') and entry.get('key'):registered.add(tuple(entry['key']))
for row in r['rows']:
 assert row['status'] in ['completed','verified_unregistered','already_completed','blocked','failed']
 if row['status']=='completed':
  k=(row['baseline'],row['operator'],row['case_id']);assert k in registered
  c=json.loads((p/row['checker_receipt']).read_text());assert c['status']=='verified';assert (c['library'],c['op'],c['case_id'])==k;v=c['verification'];assert v['checked_outputs']==v['total_outputs']>0
  row['checked_outputs']=v['checked_outputs'];row['total_outputs']=v['total_outputs'];module=row['factory'].split('.')[-2]+'.py';row['factory_sha256']=hashlib.sha256((p/module).read_bytes()).hexdigest()
  row['timings']['checker_wall_s']=c.get('wall_seconds')
  snap=p/Path(row['checker_receipt']).parent/'guard_snapshot.json'
  if snap.exists():
   s=json.loads(snap.read_text());row['timings']['queue_s']=(datetime.datetime.fromisoformat(s['acquired_at'])-datetime.datetime.fromisoformat(s['requested_at'])).total_seconds()
  if row['timings'].get('compile_s') is not None:row['timings']['compile_scope']='shared_original_target_build'
counts=collections.Counter(x['status'] for x in r['rows']);r.update(utc=now.isoformat(),completed=counts['completed'],status_counts=dict(counts));(p/'RESULT.tmp').write_text(json.dumps(r,indent=2));(p/'RESULT.tmp').replace(p/'RESULT.json')
started=datetime.datetime.fromisoformat(json.loads((p/'STARTED.json').read_text())['utc']);d=json.loads((p/'DELIVERY.json').read_text());d.update(status='delivered_with_blockers' if '--final' in sys.argv else 'in_progress',utc=now.isoformat(),completed=counts['completed'],scope_count=134,status_counts=dict(counts),mapping=[dict(key=[x['baseline'],x['operator'],x['case_id']],status=x['status'],factory=x['factory'],registered=x['registered'],source_hash=x.get('factory_sha256'),checker_receipt=x.get('checker_receipt')) for x in r['rows']],unresolved=[dict(key=[x['baseline'],x['operator'],x['case_id']],status=x['status'],reason=x['reason'],evidence=x['evidence']) for x in r['rows'] if x['status']!='completed'])
d['actual_elapsed_wall_s']=(now-started).total_seconds();d['started_at']=started.isoformat();d['source_hashes'].update({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('integration700_*.py')});d['patches']=sorted(f.name for f in p.glob('provider_init*.patch'));d['backups']=sorted(f.name for f in p.glob('provider_init*.before.py'));d['shared_changes']=sorted({*['TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'+x['factory'].split('.')[-2]+'.py' for x in r['rows'] if x['registered']], 'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_dense_graph.py','TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py exact-key registrations only'})
d['remaining']=f"{134-counts['completed']} non-completed; exact reasons and evidence in unresolved and RESULT.json."
d['quant_recipe']='npu-guard run --owner baseline700-20261026/transformer_dense --wait 3600 --timeout 300 -- /opt/tileops-all-pairs/bin/allpairs-python /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/run_formal_quant.py OP CASE_ID'
d['environment_note']='Use the formal private launch recipes to select original private OPP before torch_npu import; GELU requires installed Torch2.13 matching its unchanged original extension build. Formal launchers do not replace PROVIDERS.'
d['checks']=['EXACT_ROUTES_DEVICE_FREE.json','ALL_FACTORY_HASHES.json','SOURCE_MIGRATION_PROOF.json','YARN_MIGRATION_PROOF.json','GRAPH_ORIGIN_EVIDENCE.json','ORIGINAL_DEVICE_BINARIES.json'];d['repetition_note']='No retained successful numerical key rerun. Failed operational/binding paths were retried only after documented dependency, package provenance, dtype, scale shape, or optional-output fixes. Registration checks were device-free.'
(p/'DELIVERY.tmp').write_text(json.dumps(d,indent=2));(p/'DELIVERY.tmp').replace(p/'DELIVERY.json');print(json.dumps(dict(counts=counts,elapsed_wall_s=d['actual_elapsed_wall_s'])))
