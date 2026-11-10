import os,sys,json,tempfile,importlib.util,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');T=P/'snapshot/TileOPs'
cid,attempt=sys.argv[1:];assert cid in [r['case_id'] for r in json.loads((P/'SCOPE.json').read_text())]
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m['phase']=='running' and m['child']['pid']==os.getpgrp()
out=P/attempt;(out/'GUARD_METADATA.json').write_text(json.dumps(m,indent=2));os.chdir(out);short=Path(f'/proc/{os.getpid()}/cwd');sys.dont_write_bytecode=True
for n in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (out/n).mkdir(exist_ok=True);os.environ[n]=str(short/n)
os.environ['TMPDIR']=str(short/'tmp');tempfile.tempdir=str(short/'tmp');os.environ['BASELINE_PROGRESS_FILE']=str(out/'progress.json');os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
vendor=W/'.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn'
os.environ['OPS_NN_VENDOR']=str(vendor);os.environ['ASCEND_CLASSIC_NN_VENDOR']=str(vendor);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
sys.path[:0]=[str(T),str(T/'src'),str(W),str(P)]
import benchmarks.ascend_baselines as ab
ab.WORKSPACE=W;ab.LIBRARIES=W/'ascend_baselines'
import benchmarks.ascend_all_pairs.ops_nn as provider
from adapter import make
provider.PROVIDERS['GemmFwdOp']=make
from benchmarks.ascend_all_pairs.registry import candidate_factory
import types
probe=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFwdOp')));sentinel=object()
provider.PROVIDERS['GemmFwdOp']=lambda c:sentinel
assert candidate_factory(probe,'ops-nn')() is sentinel
provider.PROVIDERS['GemmFwdOp']=make
(out/'DISPATCH.json').write_text(json.dumps({'factory':str(P/'adapter.py')+':make','registry_dispatch_asserted':True,'provider_snapshot':str(provider.__file__),'vendor':str(vendor)}))
spec=importlib.util.spec_from_file_location('frozen_checker',T/'scripts/ascend_pair_check.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
original=checker.observe_call
# Observation only: save existing canonical tensors after completed calls; never generate or alter values.
def observe(torch,run,records):
 wrapped=original(torch,run,records)
 def capture(*args,**kwargs):
  value=wrapped(*args,**kwargs);torch.npu.synchronize()
  label='reference' if not (out/'reference.pt').exists() else 'candidate'
  def cpu(x):
   if isinstance(x,torch.Tensor):return x.detach().cpu()
   if isinstance(x,(tuple,list)):return type(x)(cpu(v) for v in x)
   if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
   return x
  if label=='reference':torch.save(cpu(args),out/'canonical_inputs.pt')
  torch.save(cpu(value),out/(label+'.pt'))
  return value
 return capture
checker.observe_call=observe
from native_log_capture import NativeLogCapture
sys.argv=[str(T/'scripts/ascend_pair_check.py'),'--op','GemmFwdOp','--case-id',cid,'--library','ops-nn','--output',str(out/'checker.json')]
with NativeLogCapture(out/'ASCEND_PROCESS_LOG_PATH',out/'native.jsonl'):rc=checker.main()
(out/'COMPLETION.json').write_text(json.dumps({'checker_returncode':rc,'sync_cleanup':'original checker finally executed; inspect checker for exceptions','finished_at':time.time()}))
raise SystemExit(rc)
