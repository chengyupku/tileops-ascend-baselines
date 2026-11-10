import json,pathlib,importlib.util,hashlib,statistics,csv
P=pathlib.Path(__file__).parent
report=json.loads((P/'RESULTS.json').read_text())
spec=importlib.util.spec_from_file_location('audit_profile',P/'v3/kernel_task_profile.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
audit=[]
for row in report['rows']:
 p=pathlib.Path(row['receipt']);m=json.loads(p.read_text());guard=json.loads((p.parent/'guard.json').read_text());checker=json.loads((p.parent/'correctness.json').read_text());env=m['identity']['environment']
 assert checker['status']=='verified' and checker['inputs_preserved']
 assert guard['performance_valid'] and guard['workload_exit_code']==0 and guard['exit_code']==0
 assert env['aicore_count']==32 and env['vectorcore_count']==64
 assert m['identity']['protocol']['id']=='ascend950-isolated-ffts-v3'
 assert len(m['timing']['latency_samples_ms'])==len(m['end_to_end_samples_ms'])==15
 assert all(m['experiment_v3_sources'][n]==hashlib.sha256((P/'v3'/n).read_bytes()).hexdigest() for n in m['experiment_v3_sources'])
 for prof in m['timing']['retained_profiles']:
  rel=pathlib.Path(prof['profile_path']).relative_to(pathlib.Path(str(p.parent).replace(str(P.resolve()),'/opt/tileops-baseline-cache/workspace/reports/baseline-v3-sample20-20261101/server1')))
  directory=p.parent/rel
  assert parser.trace_digest(directory)==prof['profile_trace_sha256']
  parsed=parser.parse_profile(directory,1)
  assert abs(parsed['latency_ms']-sum(t['duration_us'] for t in prof['profile_kernel_tasks'])/1000)<1e-6
 audit.append({'key':row['key'],'guard_valid':True,'full_checker':True,'traces_verified':15,'physical_cores':[32,64],'v3_protocol':True})
(P/'AUDIT.json').write_text(json.dumps({'accepted':len(audit),'rows':audit},indent=2))
report['mean_absolute_error_ms']={k:statistics.mean(abs(r[k+'_ms']-r['new_v3_ms']) for r in report['rows']) for k in ['old_v2','old_H','new_v2','new_H']}
report['median_absolute_relative_error']={k:statistics.median(r[k+'_absolute_relative_error'] for r in report['rows']) for k in ['old_v2','old_H','new_v2','new_H']}
report['winner_counts']={k:sum(r[k+'_absolute_relative_error']<r[other+'_absolute_relative_error'] for r in report['rows']) for k,other in [('old_v2','old_H'),('old_H','old_v2')]}
report['library_counts']={lib:sum(r['key'][0]==lib for r in report['rows']) for lib in sorted({r['key'][0] for r in report['rows']})}
report['successful_guard_hours']=sum(r['guard_elapsed_seconds'] for r in report['rows'])/3600
report['successful_profile_hours']=sum(r['measurement_wall_seconds'] for r in report['rows'])/3600
if len(audit)==20:
 assert sum(r['population_weight'] for r in report['rows'])==2814
 report['unweighted_extrapolation_hours']=report['successful_guard_hours']*2814/20
(P/'RESULTS.json').write_text(json.dumps(report,indent=2))
with (P/'COMPARISON.csv').open('w') as f:
 keys=['library','operator','case','historical_host','old_v2_ms','old_H_ms','new_v3_ms','new_v2_ms','new_H_ms','old_v2_error_pct','old_H_error_pct','guard_seconds','profile_seconds'];w=csv.writer(f);w.writerow(keys)
 for r in report['rows']:w.writerow([*r['key'],r['historical_host'],r['old_v2_ms'],r['old_H_ms'],r['new_v3_ms'],r['new_v2_ms'],r['new_H_ms'],100*r['old_v2_absolute_relative_error'],100*r['old_H_absolute_relative_error'],r['guard_elapsed_seconds'],r['measurement_wall_seconds']])
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
