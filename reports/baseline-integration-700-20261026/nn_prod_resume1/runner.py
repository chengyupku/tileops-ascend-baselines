import os,sys,json,time,tempfile,hashlib,threading,shutil
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');T=W/'TileOPs'
sys.dont_write_bytecode=True
m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert m['phase']=='running' and m['child']['pid']==os.getpgrp();assert m['owner']=='baseline700-20261026/nn_prod_resume1'
(P/'GUARD_METADATA.json').write_text(json.dumps(m,indent=2));os.chdir(P)
for n in ['tmp','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 (P/n).mkdir(exist_ok=True);os.environ[n]=str(Path(f'/proc/{os.getpid()}/cwd')/n)
os.environ['TMPDIR']=str(P/'tmp');tempfile.tempdir=str(P/'tmp');os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0';os.environ['BASELINE_PROGRESS_FILE']=str(P/'progress.json')
vendor=W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'
for n in ['ASCEND_CLASSIC_NN_VENDOR','OPS_NN_VENDOR','ASCEND_CUSTOM_OPP_PATH']:os.environ[n]=str(vendor)
sys.path[:0]=[str(P),str(T),str(T/'src'),str(W)]
from benchmarks.ascend_all_pairs import ops_nn as provider
from benchmarks.ascend_all_pairs.ops_nn import graph
# Only relocate the graph cache; native source/compiler/options remain unchanged.
graph.WORKSPACE=P
from adapter import make
provider.PROVIDERS['ProdFwdOp']=make
from benchmarks.ascend_all_pairs.registry import candidate_factory
import types,inspect
probe=types.SimpleNamespace(id='long-seq-bfloat16',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='ProdFwdOp')));sentinel=object()
provider.PROVIDERS['ProdFwdOp']=lambda c:sentinel if c is probe else None
assert candidate_factory(probe,'ops-nn')() is sentinel
provider.PROVIDERS['ProdFwdOp']=make
(P/'DISPATCH.json').write_text(json.dumps({'factory':str(P/'adapter.py')+':make','registry_dispatch_asserted':True,'cache_only_relocation':str(P)}))
source=T/'scripts/ascend_pair_check.py';frozen=P/'frozen/TileOPs/scripts/ascend_pair_check.py'
assert source.read_bytes()==frozen.read_bytes()
ns={'__name__':'frozen_checker','__file__':str(source)};exec(compile(frozen.read_bytes(),str(source),'exec'),ns)
original_observe=ns['observe_call'];counter=0
# Passive evidence snapshots around each canonical call; no extra candidate calls.
def observe(torch,run,records):
 global counter
 label='call'+str(counter);counter+=1
 observed=original_observe(torch,run,records)
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
  if isinstance(x,(tuple,list)):return type(x)(cpu(t) for t in x)
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  return x
 def wrapped(*args,**kwargs):
  before=cpu((args,kwargs));result=observed(*args,**kwargs)
  torch.save({'inputs_before':before,'inputs_after':cpu((args,kwargs)),'outputs':cpu(result)},P/(label+'.pt'))
  return result
 return wrapped
ns['observe_call']=observe
from native_log_capture import NativeLogCapture
# Preserve raw log bytes through rotations and any transient native ELF.
stop=threading.Event();seen={};elfs={};(P/'rawlogs').mkdir(exist_ok=True);(P/'elf').mkdir(exist_ok=True)
def capture():
 while not stop.is_set():
  for root in [P/'ASCEND_PROCESS_LOG_PATH',P/'tmp',P/'ASCEND_CACHE_PATH',P/'.cache']:
   for f in root.rglob('*'):
    try:
     if not f.is_file():continue
     st=f.stat()
     if f.suffix=='.log':
      key=(str(f),st.st_ino);off=seen.get(key,0)
      if st.st_size<off:off=0
      with f.open('rb') as src:src.seek(off);data=src.read()
      if data:
       name=hashlib.sha256(str(key).encode()).hexdigest()+'.log'
       with (P/'rawlogs'/name).open('ab') as dest:dest.write(data)
      seen[key]=off+len(data)
     elif f.suffix in ('.o','.so','.bin') and (str(f),st.st_size,st.st_mtime_ns) not in elfs:
      data=f.read_bytes()
      if data.startswith(b'\x7fELF'):
       h=hashlib.sha256(data).hexdigest();(P/'elf'/h).write_bytes(data);elfs[(str(f),st.st_size,st.st_mtime_ns)]=h
    except (FileNotFoundError,PermissionError):pass
  stop.wait(.05)
thread=threading.Thread(target=capture);thread.start();start=time.time();rc=1
try:
 sys.argv=[str(source),'--op','ProdFwdOp','--case-id','long-seq-bfloat16','--library','ops-nn','--output',str(P/'checker.json')]
 with NativeLogCapture(P/'ASCEND_PROCESS_LOG_PATH',P/'native.jsonl'):rc=ns['main']()
finally:
 stop.set();thread.join()
 (P/'ELF_MANIFEST.json').write_text(json.dumps([{'source':k[0],'sha256':v} for k,v in elfs.items()],indent=2))
 (P/'CHILD_RESULT.json').write_text(json.dumps({'returncode':rc,'elapsed_s':time.time()-start}))
sys.exit(rc)
