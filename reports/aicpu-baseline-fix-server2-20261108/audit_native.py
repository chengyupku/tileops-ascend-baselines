import pathlib,json,hashlib,importlib.util,statistics,csv,datetime
P=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/opt/torch-all2208-20261105')
spec=importlib.util.spec_from_file_location('parser',B/'tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
oldrows={tuple(r[k] for k in ['library','op','case']):r for r in json.load(open(P/'BACKLOG.json'))};rows=[];windows=0
tasks=[f for f in (P/'native-v2/runs').glob('*/a1/task.json') if f.parent.parent.name!='54ad3ceb5b25198d019c']+list((P/'native-v4/runs').glob('*/a1/task.json'))
assert len(tasks)==10
for task in sorted(tasks):
 out=task.parent;m=json.load(open(out/'measurement.json'));g=json.load(open(out/'guard.json'));old=oldrows[tuple(m['key'])]
 assert m['status']=='measured_pending_guard' and g['performance_valid'] and g['workload_exit_code']==0
 assert datetime.datetime.fromisoformat(g['acquired_at']).timestamp() <= m['started_at'] <= m['finished_at'] <= datetime.datetime.fromisoformat(g['updated_at']).timestamp()
 if (out/'group_member.json').exists():
  member=json.load(open(out/'group_member.json'));assert member['key']==m['key'] and member['exit_code']==0
  assert member['started_at'] <= m['started_at'] <= m['finished_at'] <= member['finished_at']
 c=json.load(open(out/'correctness.json'));assert c['status']=='verified' and c['inputs_preserved']
 edge=json.load(open(out/'raw_bool_check.json'));assert edge['status']=='verified' and edge['exact']
 assert m['correctness_sha256']==sha(out/'correctness.json')
 t=m['timing'];assert len(t['retained_profiles'])==15;samples=[];aicpu=[]
 for prof in t['retained_profiles']:
  path=pathlib.Path(prof['profile_path']);assert parser.trace_digest(path)==prof['profile_trace_sha256'];parsed=parser.parse_profile(path,1)
  assert parsed['profile_kernel_tasks']==prof['profile_kernel_tasks'] and parsed['profile_named_task_coverage_complete'];samples.append(parsed['latency_ms']);aicpu.append(parsed['profile_aicpu_task_count']);windows+=1
 assert all(v==0 for v in aicpu), (m['key'],aicpu)
 assert abs(statistics.mean(samples)-t['latency_ms'])<1e-9
 for n,h in m['identity']['sources'].items():
  p=pathlib.Path(n);p=p if p.is_absolute() else pathlib.Path('/opt/tileops-all-pairs/workspace')/p;assert sha(p)==h,n
 rows.append({'library':m['key'][0],'op':m['key'][1],'case':m['key'][2],'old_web_us':float(old['latency_us']),'fixed_us':t['latency_ms']*1000,'baseline_improvement_vs_published':float(old['latency_us'])/(t['latency_ms']*1000),'old_web_ratio':float(old['ratio']),'recomputed_ratio_with_published_tileops':t['latency_ms']*1000/float(old['tileops_latency_us']),'tileops_published_us':float(old['tileops_latency_us']),'fixed_aicpu':0,'measurement_sha256':sha(out/'measurement.json'),'guard_sha256':sha(out/'guard.json')})
(P/'NATIVE_RESULTS.json').write_text(json.dumps({'status':'verified','pairs':len(rows),'raw_windows':windows,'rows':rows,'comparison_note':'New server2 measurement vs published old server1 measurement; same32AIC/64AIV protocol, not a same-round native A/B.'},indent=2))
with (P/'NATIVE_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'pairs':len(rows),'windows':windows,'status':'verified'}))
