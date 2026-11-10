"""CPU reparse all retained raw windows, SHA checks and same-metric comparison."""
import pathlib,json,hashlib,importlib.util,statistics,csv
P=pathlib.Path(__file__).resolve().parent;B=pathlib.Path('/opt/torch-all2208-20261105')
spec=importlib.util.spec_from_file_location('parser',B/'tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.load(open(P/'SOURCE_PROOF.json'))
for n,h in proof.items():
 p=P/n.removeprefix('candidate/') if n.startswith('candidate/') else B/n
 assert sha(p)==h,('source changed',n)
guard=json.load(open(P/'formal14-guard.json'))
assert guard['performance_valid'] and guard['workload_exit_code']==0 and guard['phase']=='finished'
rows=[];windows=0
for old in json.load(open(P/'TASKS.json')):
 out=P/'runs'/(old['op']+'--'+old['case']);data={}
 for variant in ['original','fixed']:
  f=out/variant/'measurement.json';m=json.load(open(f));t=m['timing'];samples=[];aicpu=[]
  assert len(t['retained_profiles'])==15
  for prof in t['retained_profiles']:
   path=pathlib.Path(prof['profile_path']);assert parser.trace_digest(path)==prof['profile_trace_sha256']
   parsed=parser.parse_profile(path,1)
   assert parsed['profile_kernel_tasks']==prof['profile_kernel_tasks']
   assert parsed['profile_named_task_coverage_complete']
   samples.append(parsed['profile_kernel_us']/1000);aicpu.append(parsed['profile_aicpu_task_count']);windows+=1
  assert abs(statistics.mean(samples)-t['latency_ms'])<1e-9
  assert all(abs(a-b)<1e-9 for a,b in zip(samples,t['latency_samples_ms']))
  data[variant]={'us':1000*t['latency_ms'],'aicpu_count_per_window':aicpu,'measurement_sha256':sha(f)}
 assert all(v==0 for v in data['fixed']['aicpu_count_per_window'])
 rows.append({'library':old['library'],'op':old['op'],'case':old['case'],'old_web_us':float(old['latency_us']),'same_round_original_us':data['original']['us'],'fixed_us':data['fixed']['us'],'baseline_improvement':data['original']['us']/data['fixed']['us'],'old_web_ratio':float(old['ratio']),'recomputed_ratio_with_published_tileops':data['fixed']['us']/float(old['tileops_latency_us']),'tileops_published_us':float(old['tileops_latency_us']),'original_measurement_sha256':data['original']['measurement_sha256'],'fixed_measurement_sha256':data['fixed']['measurement_sha256'],'fixed_aicpu':0})
result={'status':'verified','pairs':len(rows),'raw_windows':windows,'guard_sha256':sha(P/'formal14-guard.json'),'source_proof_sha256':sha(P/'SOURCE_PROOF.json'),'rows':rows,'ratio_note':'Uses existing published TileOPs denominator; TileOPs not remeasured in this baseline-only task.'}
(P/'TORCH_RESULTS.json').write_text(json.dumps(result,indent=2))
with (P/'TORCH_RESULTS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'pairs':len(rows),'windows':windows,'status':'verified'},indent=2))
