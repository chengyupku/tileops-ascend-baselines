"""Bind pinned original PTO public interface to the actual old1076 guard."""
import dataclasses,hashlib,importlib.util,json,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=Path(sys.argv[sys.argv.index('--out')+1]).resolve()
task=json.loads(Path(sys.argv[sys.argv.index('--task')+1]).read_text())
expected='baseline-cache-20261027/server1/'+task['id']
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
if meta.get('owner')!=expected or meta.get('phase')!='running' or meta.get('child',{}).get('pid')!=os.getpgrp():raise RuntimeError('Actual exact-key foreground guard required')
os.environ['PTO23_OWNER']=expected
os.environ['OLD1076_PAIR_OUT']=str(out)
runroot=out/'pto_run';runroot.mkdir(exist_ok=False)
os.environ['PTO23_RUN_ROOT']=str(runroot)
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
path=ROOT/'tools/baseline_old1076_pto_registration_v1.py'
spec=importlib.util.spec_from_file_location('old1076_pto_registration',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.DATA_ROOT=ROOT/'reports/baseline-integration-700-20261026/threehost_pto_registration23'
module.require_guard()
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
 if library!='tilelang-ascend' or case._call.signature.name not in ('BmmFwdOp','GemmFwdOp'):raise ValueError('PTO GEMM/BMM only')
 def build():
  c=module.make(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__),path)},adaptation=c.adaptation+' Current exact-key guard owner and private output directory bound at top interface; original native/compiler pins unchanged.')
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
