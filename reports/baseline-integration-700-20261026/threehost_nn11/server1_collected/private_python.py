import runpy,sys
from pathlib import Path
sys.dont_write_bytecode=True
W=Path('/opt/tileops-all-pairs/workspace')
sys.path[:0]=[str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),'/usr/local/Ascend/cann-9.1.0/python/site-packages','/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe']
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
a=sys.argv[1:]
if not a:raise SystemExit('Usage: allpairs-python -c CODE | -m MODULE | SCRIPT')
if a[0]=='-c':
 sys.argv=['-c']+a[2:];exec(compile(a[1],'<command>','exec'),{'__name__':'__main__'})
elif a[0]=='-m':
 sys.argv=a[1:];runpy.run_module(a[1],run_name='__main__',alter_sys=True)
else:
 p=Path(a[0]).resolve();sys.argv=[str(p)]+a[1:];sys.path.insert(0,str(p.parent));runpy.run_path(str(p),run_name='__main__')
