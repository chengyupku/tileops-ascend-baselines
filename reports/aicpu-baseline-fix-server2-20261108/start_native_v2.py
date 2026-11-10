"""Foreground dependency gate; no device reservation during CPU waiting."""
import pathlib,json,time,subprocess,sys
P=pathlib.Path(__file__).resolve().parent;deadline=time.monotonic()+900
while True:
 try:g=json.load(open(P/'mask-v2-guard.json'))
 except (FileNotFoundError,json.JSONDecodeError):g={}
 if g.get('phase') in ('finished','released_without_launch'):
  assert g.get('workload_exit_code')==0 and g.get('performance_valid'),g
  rows=json.load(open(P/'MASK_EDGE_CHECKS_V2.json'));assert all(r['all256_patterns_exact'] for r in rows)
  break
 if time.monotonic()>deadline:raise RuntimeError('mask prerequisite unavailable; native batch not started')
 time.sleep(1)
raise SystemExit(subprocess.run([sys.executable,str(P/'run_native_v2.py')]).returncode)
