"""CPU-only binary trace replay of completed guarded measurements."""
import pathlib,json,hashlib,importlib.util,math
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('raw_parser',P/'tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
result=P/'RAW_REPLAY.json';rows=json.loads(result.read_text()) if result.exists() else []
known={r['measurement_sha256'] for r in rows}
for f in sorted((P/'runs').glob('*/*/measurement.json')):
 digest=hashlib.sha256(f.read_bytes()).hexdigest()
 if digest in known:continue
 guards=[]
 for g in f.parent.glob('guard*.json'):
  try:v=json.loads(g.read_text())
  except json.JSONDecodeError:continue
  if v.get('phase')=='finished' and v.get('performance_valid') and v.get('workload_exit_code')==0:guards.append(g)
 if not guards:continue
 m=json.loads(f.read_text());t=m['timing'];windows=[]
 for recorded,sample in zip(t['retained_profiles'],t['latency_samples_ms'],strict=True):
  path=recorded['profile_path'];h=parser.trace_digest(path);assert h==recorded['profile_trace_sha256']
  parsed=parser.parse_profile(path,1);assert math.isclose(parsed['latency_ms'],sample,abs_tol=1e-7)
  assert parsed['profile_kernel_tasks']==recorded['profile_kernel_tasks']
  windows.append({'trace_sha256':h,'latency_ms':parsed['latency_ms'],'aicpu_tasks':parsed['profile_aicpu_task_count']})
 assert len(windows)==15
 rows.append({'ordinal':int(f.parent.parent.name),'version':f.parent.name,'measurement_sha256':digest,'windows':windows,'status':'15 raw windows exactly match recorded tasks and timing'})
 result.write_text(json.dumps(rows,indent=2));print(f.parent.relative_to(P),flush=True)
print('Replayed measurements:',len(rows),flush=True)
