import pathlib,subprocess,json,time
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="""import pathlib,json
p=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tla_other54_r2');rows=[]
for f in (p/'runs').glob('*/checker.json'):
 d=json.loads(f.read_text());loads=[]
 for n in f.parent.glob('native_load*json'):
  l=json.loads(n.read_text());l.pop('maps',None);loads.append(l)
 rows.append(dict(run=f.parent.name,checker=d,native_loads=loads,guard=json.loads((f.parent/'GUARD_RUNNING.json').read_text()),dispatches=len(list(f.parent.glob('*.dispatch.json')))))
print(json.dumps(rows))
"""
r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);assert r.returncode==0,r.stderr
(P/'collected_checkers.json').write_text(r.stdout)
result=json.loads((P/'RESULT.json').read_text())
for v in json.loads(r.stdout):
 c=v['checker'];row=next(x for x in result['rows'] if [x['baseline'],x['operator'],x['case_id']]==[c['library'],c['op'],c['case_id']]);row.update(status='verified_unregistered' if c['status']=='verified' else c['status'],reason='Official release SDK headers resolve original Cube ABI; canonical checker passed.' if c['status']=='verified' else c.get('error'),numerical_success_this_attempt=c['status']=='verified',device_rerun=True,checker_receipt='remote server1 runs/'+v['run']+'/checker.json',verification=c.get('verification'),inputs_preserved=c.get('inputs_preserved'),native_loads=v['native_loads'],native_evidence_complete=bool(v['native_loads']),native_gap=None if v['native_loads'] else 'TVM-FFI dispatch/maps captured; actual symbol pointer and loaded exported.so bytes were not captured before process exit. No successful case rerun.',checker_wall_seconds=c.get('wall_seconds'),guard_metadata=v['guard'])
result['updated']=time.time();result['numerically_verified_this_worker']=sum(x['numerical_success_this_attempt'] for x in result['rows']);(P/'RESULT.json').write_text(json.dumps(result,indent=2));print('verified',result['numerically_verified_this_worker'])
