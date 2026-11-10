"""Private exactkey provider and unchanged original fullchecker, no registration."""
import os,sys,pathlib,tempfile,json,hashlib,runpy
R=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
KEY=sys.argv[sys.argv.index('--case-id')+1]
OP=sys.argv[sys.argv.index('--op')+1]
assert '/' not in KEY
os.chdir(R);(R/'tmp').mkdir(exist_ok=True)
tempfile.tempdir=f'/proc/{os.getpid()}/cwd/tmp';os.environ['TMPDIR']=tempfile.tempdir
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT),str(R)]
FIRST='gpt-oss-sink-decode-bfloat16-bfloat16'
if KEY!=FIRST:
 gate=json.loads((R/(FIRST+'.checker.json')).read_text())
 assert gate['status']=='verified' and gate['inputs_preserved'] and gate['verification']['checked_outputs']==gate['verification']['total_outputs']>0
 assert (R/(FIRST+'.exit')).read_text().strip()=='0'
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import Candidate
from benchmarks.api import Implementation
SOURCE=R.parent/'attention_native_build5/attempt03/stage/source'
files=[R/'case_runner.py',R/'firstcase_adapter.py',R/'remaining_adapter.py',SOURCE/'attention/flash_attn/op_kernel/flash_attn.cpp',SOURCE/'attention/flash_attn/torch_extension/flash_attn.py',SOURCE/'attention/flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata_aicpu.cpp']
PINS={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
launch=[]
def factory(case,library):
 assert library=='ops-transformer' and case.id==KEY and case._call.signature.name==OP
 if KEY==FIRST:
  assert OP=='GQAPagedFwdOp'
  from firstcase_adapter import firstcase
  original=firstcase
 else:
  from remaining_adapter import build_run
  original=build_run(case)
 def run(*args):
  output=original(*args)
  import torch
  torch.npu.synchronize()
  launch.append({'api':'original flash_attn_metadata + flash_attn','returned_and_synchronized':True})
  return output
 return lambda:Candidate(Implementation(run),'cann_ops_transformer.flash_attn_metadata + flash_attn','Original metadata and native attention; constant compressed mask when required; unchanged inputs/sinks; private process registry',dict(PINS))
registry.candidate_factory=factory
try:
 runpy.run_path(str(ROOT/'TileOPs/scripts/ascend_pair_check.py'),run_name='__main__')
finally:
 maps=pathlib.Path('/proc/self/maps').read_text();(R/(KEY+'.maps')).write_text(maps)
 paths=set(line.split()[-1] for line in maps.splitlines() if '/' in line and any(s in line for s in ['.so','kernel','cann_ops']))
 pins={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sorted(paths) if pathlib.Path(p).is_file()}
 loaded_python={name:{'path':str(pathlib.Path(mod.__file__).resolve()),'sha256':hashlib.sha256(pathlib.Path(mod.__file__).read_bytes()).hexdigest()} for name,mod in sys.modules.items() if name.startswith('cann_ops_transformer') and getattr(mod,'__file__',None) and pathlib.Path(mod.__file__).is_file()}
 (R/(KEY+'.native.json')).write_text(json.dumps({'launch':launch,'loaded_python':loaded_python,'loaded_files_sha256':pins,'exact_kernel_binary':'pending runtime-log attribution; no numerical acceptance claim','ASCEND_CUSTOM_OPP_PATH':os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'private_registry':True},indent=2))
