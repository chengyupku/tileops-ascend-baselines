from pathlib import Path
import json,hashlib,datetime
T=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((T/'FROZEN.json').read_text());changed=[p for p,h in frozen['sha256'].items() if sha(Path(p))!=h]
conv_frozen=json.loads((T/'FROZEN_CONV.json').read_text());assert all(sha(Path(p))==h for p,h in conv_frozen['sha256'].items())
r=json.loads((T/'RESULT.json').read_text());details=[]
for row in r['rows']:
 P=Path(row.get('evidence',str(T/row['key'][2])));d=dict(row)
 if (P/'GUARD_RECEIPT.json').exists():
  g=json.loads((P/'GUARD_RECEIPT.json').read_text());d['guard']=g
  snapshots=g['guard_samples']
  if (P/'guard_snapshot.json').exists():snapshots=snapshots+[json.loads((P/'guard_snapshot.json').read_text())]
  running=next((x for x in snapshots if x.get('phase')=='running'),None)
  if running:
   ts=lambda x:datetime.datetime.fromisoformat(x).timestamp()
   d['timing_summary']={'requested_at':running['requested_at'],'acquired_at':running['acquired_at'],'started_at':running['updated_at'],'end_observed_at':g['end'],'queue_seconds':ts(running['acquired_at'])-ts(running['requested_at']),'guard_command_wall_seconds_upper_bound':ts(g['end'])-ts(running['updated_at'])}
 if (P/'TIMINGS.json').exists():d['observed_call_seconds']=json.loads((P/'TIMINGS.json').read_text())
 if (P/'PHASE.json').exists():d['last_phase']=json.loads((P/'PHASE.json').read_text())
 if (P/'representative.json').exists():
  c=json.loads((P/'representative.json').read_text());d['canonical']={k:c.get(k) for k in ['status','input_seed','verification','reference_device','independent_cpu_reference','oracle_source_sha256','oracle_source_sha256_after','provider_source_sha256','provider_source_sha256_after','error']}
 if (P/'CAPTURE.json').exists():
  cap=json.loads((P/'CAPTURE.json').read_text());d['output_proof']={k:v for k,v in cap.items() if 'output' in k}
  if row['key'][1]=='GroupedGemmFwdOp':d['group_metadata']={k:v for k,v in cap.items() if k in ['candidate_0_input2','candidate_0_input3']}
 if (P/'NATIVE_DISPATCH.log').exists():
  lines=(P/'NATIVE_DISPATCH.log').read_text().splitlines();d['native_api_call_counts']={'workspace':sum('Entering function aclnnGroupedMatmulV5GetWorkspaceSize' in x for x in lines),'launch':sum(x.rstrip().endswith('Entering function aclnnGroupedMatmulV5') for x in lines)}
  if row['status']=='verified_unregistered':
   expected=8 if row['key'][2]=='qwen3-30b-ep8-empty-bfloat16' else 16 if row['key'][1]=='GroupedGemmFwdOp' else 1
   assert d['native_api_call_counts']=={'workspace':expected,'launch':expected}
 d['evidence_directory']=str(P);details.append(d)
out={'accepted_added':0,'registered':False,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete_scoped_execution_pending_parent_acceptance','verified_unregistered':sum(x['status']=='verified_unregistered' for x in details),'blocked':sum(x['status']=='blocked' for x in details),'pins_checked_after':len(frozen['sha256']),'changed_frozen_paths':changed,'deployment':frozen['release'],'rows':details,'conv_harness_correction':json.loads((T/'CONV_PREFLIGHT_CORRECTION.json').read_text()),'conv_pins_checked_after':len(conv_frozen['sha256']),'limits':['No formal registry writes; parent acceptance pending.','Call timings include Python/layout/copies and original CPU reference; not a benchmark.','Guard end timestamps are supervisory observations (5 second polling).','Unexecuted dependent keys are blocked, never numerical failures.']}
assert not changed,changed
(T/'DELIVERY.json').write_text(json.dumps(out,indent=2))
(T/'HEARTBEAT.json').write_text(json.dumps({'at':out['at'],'phase':'complete','accepted_added':0,'verified_unregistered':out['verified_unregistered'],'blocked':out['blocked']},indent=2))
files={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'}
(T/'FILEHASHES.json').write_text(json.dumps(files,indent=2));print(json.dumps({k:out[k] for k in ['verified_unregistered','blocked','pins_checked_after','changed_frozen_paths']}))
