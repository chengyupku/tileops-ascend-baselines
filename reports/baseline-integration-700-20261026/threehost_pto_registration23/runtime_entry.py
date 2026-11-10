"""Run only inside the foreground host guard; installs the overlay in this process."""
import sys,os,json,runpy,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import registration as M
M.require_guard()
host,pin=M.current_host()
op,cid=sys.argv[1:3]
if (op,cid) not in {(r['op'],r['case_id']) for r in M.rows()}:raise RuntimeError('Unknown exact key')
root=P/'runs';root.mkdir(exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='r',dir=root));os.environ['PTO23_RUN_ROOT']=str(out)
base=Path(pin['base']);W=Path(pin['workspace']);V=Path(pin['vendor']);C=Path(pin['compiler'])
M.checked(P/('setup_'+host+'.py'),M.dataset('LOCAL_PINS.json')['setup_'+host+'.py'])
M.load('_pto23_launch_setup',P/('setup_'+host+'.py')).setup(base,W,V,C,base/'runtime',out)
from benchmarks.ascend_all_pairs import tla
M.install(tla.PROVIDERS)
checker=base/'frozen/scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','tilelang-ascend','--output',str(out/'checker.json')]
try:runpy.run_path(str(checker),run_name='__main__')
finally:
 torch=sys.modules.get('torch')
 if torch is not None and torch.npu.is_initialized():torch.npu.synchronize()
