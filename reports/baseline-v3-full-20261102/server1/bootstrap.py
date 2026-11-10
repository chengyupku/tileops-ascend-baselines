"""Experiment-only measurement module injection; unchanged canonical checker/native callables."""
import sys,os,importlib.util,json,time,hashlib,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/opt/tileops-baseline-cache/workspace')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import tools
for name in ('baseline_protocol','kernel_task_profile','measure_baselines'):
 spec=importlib.util.spec_from_file_location('tools.'+name,HERE/'v3'/f'{name}.py')
 mod=importlib.util.module_from_spec(spec);sys.modules['tools.'+name]=mod;setattr(tools,name,mod);spec.loader.exec_module(mod)
core=sys.modules['tools.measure_baselines'];core.ROOT=ROOT;core.TILEOPS_ROOT=ROOT/'TileOPs'
out=Path(sys.argv[sys.argv.index('--out')+1]);os.environ['TILEOPS_PROFILE_DIR']=str(out/'profiles')
original=core._time_reference
def measured(*args,**kwargs):
 start=time.monotonic();r=original(*args,**kwargs);r['measurement_wall_seconds']=time.monotonic()-start;return r
core._time_reference=measured
start=time.monotonic()
try:runpy.run_path(str(ROOT/'tools/measure_baseline_retest32_server1.py'),run_name='__main__')
finally:
 p=out/'measurement.json'
 if p.exists():
  r=json.loads(p.read_text());r['experiment_wall_seconds']=time.monotonic()-start;r['experiment_v3_sources']={n:hashlib.sha256((HERE/'v3'/n).read_bytes()).hexdigest() for n in ['baseline_protocol.py','kernel_task_profile.py','measure_baselines.py']};p.write_text(json.dumps(r,indent=2))
