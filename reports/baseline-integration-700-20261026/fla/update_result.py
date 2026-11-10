import json,pathlib,datetime
p=pathlib.Path(__file__).resolve().parent
registered=set()
for f in p.glob('registration.*.json'):
 registered.update(tuple(k) for k in json.loads(f.read_text())['keys'])
rows={}
for f in sorted(p.glob('*.checker.json'),key=lambda f:f.stat().st_mtime):
 d=json.loads(f.read_text());v=d.get('verification',{});passed=d.get('status')=='verified';key=(d['op'],d['case_id'])
 if f.name=='CumsumFwdOp.checker.json':continue
 reg=key in registered and passed
 guard='probe.guard.log' if f.name in ['AddFwdOp.checker.json','GemmFwdOp.checker.json','GDNFwdOp.checker.json','KDAFwdOp.checker.json'] else 'batch2.guard.log'
 if f.name=='CumsumFwdOp.explicit.checker.json' or f.name.startswith('GDNFwdOp.'):guard='batch3.guard.log' if f.name!='GDNFwdOp.checker.json' else 'probe.guard.log'
 if '.repair.' in f.name:guard='batch4.guard.log'
 for command in sorted(p.glob('*.COMMAND.json'),key=lambda f:f.stat().st_mtime):
  if d['case_id'] in json.loads(command.read_text()):guard=command.name.replace('.COMMAND.json','.guard.log')
 rows[key]=dict(baseline=d['library'],operator=d['op'],case_id=d['case_id'],status=('completed' if reg else 'verified_unregistered') if passed else 'failed',reason=d.get('error') or 'Full canonical checker passed; no benchmark.',api=d.get('api'),guard_receipt=str(p/guard),checker_receipt=str(f),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',2 if d['op'] in ['GDNFwdOp','GLAFwdOp','DeltaNetFwdOp','KDAFwdOp'] else 1),registered=reg,factory='benchmarks.ascend_all_pairs.fla.integration700_fla_native.build' if reg else None,changed_files=['integration700_fla_native.py'],timings=dict(queue=None,device=None,compile=None,checker_wall=d.get('wall_seconds')),source_hash=d.get('source_sha256'))
if (p/'decode_blockers.json').exists():
 for r in json.loads((p/'decode_blockers.json').read_text())['rows']:
  rows.setdefault((r['operator'],r['case_id']),r)
for r in rows.values():
 r['changed_files']=[str(p/'integration700_fla_native.py')]
 if r['registered']:r['changed_files'] += [str(p.parents[2]/'TileOPs/benchmarks/ascend_all_pairs/fla/integration700_fla_native.py'),str(p.parents[2]/'TileOPs/benchmarks/ascend_all_pairs/fla/__init__.py')]
 if r['operator']=='KDAFwdOp' and 'DT_DOUBLE' in r['reason']:
  r['status']='blocked'
  r['reason']='Canonical oracle blocked before candidate: '+r['reason']
  r['proposed_parent_patch']=str(p/'PROPOSED_KDA_checker.patch')
 guard=pathlib.Path(r['guard_receipt']); structured=guard.with_name(guard.name.replace('.guard.log','.GUARD_RESULT.json'))
 if structured.exists():
  r['guard_receipt']=str(structured)
  g=json.loads(structured.read_text())
  r['timings'].update(queue=g.get('queue_seconds'),device=g.get('device_seconds'))
(p/'RESULT.json.tmp').write_text(json.dumps(dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=list(rows.values())),indent=2));(p/'RESULT.json.tmp').replace(p/'RESULT.json')
print({s:sum(r['status']==s for r in rows.values()) for s in set(r['status'] for r in rows.values())})
