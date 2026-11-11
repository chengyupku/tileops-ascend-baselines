"""Guarded first-batch measurement; acceptance requires external guard and provenance audit."""
import argparse,ctypes,dataclasses,fcntl,hashlib,json,os,pathlib,sys,time,traceback

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--windows',type=int,choices=[1,15],required=True);p.add_argument('--claim',type=pathlib.Path,required=True);p.add_argument('--guard-lock',type=pathlib.Path,required=True);p.add_argument('--flush-dir',type=pathlib.Path,required=True);p.add_argument('--flush-sha256',required=True);p.add_argument('--flush-build-label',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 # Require a held host reservation; the caller runs this whole foreground process under guard.
 with a.guard_lock.open() as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('No held host device reservation')
 root=a.root;s=root/'frozen/source';sys.path[:0]=[str(s/'src'),str(s)]
 report={'status':'running','kind':'first_batch_measurement_pending_guard_and_environment_acceptance'}
 try:
  import torch,torch_npu
  from benchmarks.api import cases
  from workloads.device import set_run_device
  from workloads.numerics import compare_outputs,describe
  torch.set_num_threads(2);torch.manual_seed(0);set_run_device('cpu')
  row=json.loads(a.claim.read_text())['rows'][a.index]
  c=next(c for c in cases(row['op']) if c.id==row['case_id'])
  assert json.loads(json.dumps(c.params))==row['case']['params'], 'frozen case params mismatch'
  xs=c.inputs
  expected=c.reference(*tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in xs))
  torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
  def cv(x,device):
   if isinstance(x,torch.Tensor):return x.to(device)
   if isinstance(x,tuple):return tuple(cv(v,device) for v in x)
   if isinstance(x,list):return [cv(v,device) for v in x]
   if isinstance(x,dict):return {k:cv(v,device) for k,v in x.items()}
   return x
  args=cv(xs,'npu');set_run_device('npu')
  report.update(op=row['op'],case=row['case_id'],contract_sha256=row['contract_sha256'],torch=torch.__version__,torch_npu=torch_npu.__version__,compiler_backend=os.environ.get('TORCHINDUCTOR_NPU_BACKEND'),cube=32,vector=64,verification=describe(c.verification))
  report['phase']='eager_correctness';out=c.reference(*args);torch.npu.synchronize();eager_check=compare_outputs(cv(out,'cpu'),expected,c.verification);assert eager_check.checked_outputs==eager_check.total_outputs and eager_check.checked_outputs>0;report['eager_check']=dataclasses.asdict(eager_check);report['eager_correctness']='passed'
  report['phase']='compile_first_call';torch._dynamo.config.suppress_errors=False
  fn=torch.compile(c.reference,backend='inductor',fullgraph=True,dynamic=False)
  t=time.monotonic();out=fn(*args);torch.npu.synchronize();report['cold_first_call_seconds']=time.monotonic()-t
  report['phase']='compile_correctness';compile_check=compare_outputs(cv(out,'cpu'),expected,c.verification);assert compile_check.checked_outputs==compile_check.total_outputs and compile_check.checked_outputs>0;report['compile_check']=dataclasses.asdict(compile_check);report['compile_correctness']='passed'
  report['graph_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
  sys.path.insert(0,str(root/'frozen'))
  from tools import kernel_task_profile_full32 as full32
  sys.modules['tools.kernel_task_profile']=full32
  from tools import measure_baselines as core
  report['phase']='flush_validation'
  fd=a.flush_dir
  binary_hash=hashlib.sha256((fd/'flush.aibin').read_bytes()).hexdigest()
  assert binary_hash==a.flush_sha256
  dll=ctypes.CDLL(str(fd/'flush.so'))
  dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int
  dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
  assert dll.init(str(fd/'flush.aibin').encode())==0
  def exported_flush(buffer):
   assert dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream)==0
  core._flush_kernel_cache[0]=(exported_flush,torch.empty(268435456,dtype=torch.int32,device='npu'))
  report['flush_build']={'sha256':binary_hash,'compiler':a.flush_build_label,'historical_byte_identical':False}
  report['windows']=a.windows
  flush,buffer=core._get_flush_kernel_and_buffer(torch,torch_npu)
  assert buffer.numel()*buffer.element_size()==1024**3
  buffer.fill_(1);flush(buffer);torch.npu.synchronize()
  assert torch.count_nonzero(buffer.cpu()).item()==0, 'flush did not zero entire 1GiB buffer'
  report['flush_bytes']=1024**3
  report['timings']={}
  for variant,call in [('eager',c.reference),('compile',fn)]:
   report['phase']='timing_'+variant
   os.environ['TILEOPS_PROFILE_DIR']=str(a.output/variant/'profiles')
   last={}
   def capture(*values):
    value=call(*values);last['output']=value;return value
   timing=core._time_reference(torch,torch_npu,capture,args,warmup=3,repeats=a.windows)
   check=compare_outputs(cv(last['output'],'cpu'),expected,c.verification)
   assert check.checked_outputs==check.total_outputs and check.checked_outputs>0
   report.setdefault('timed_checks',{})[variant]=dataclasses.asdict(check)
   samples=[]
   for prof in timing['retained_profiles']:
    assert full32.trace_digest(prof['profile_path'])==prof['profile_trace_sha256']
    parsed=full32.parse_profile(pathlib.Path(prof['profile_path']),1)
    assert parsed['profile_named_task_coverage_complete']
    assert parsed['profile_kernel_tasks']==prof['profile_kernel_tasks']
    samples.append(parsed['profile_kernel_us']/1000)
   assert len(samples)==a.windows
   assert all(abs(x-y)<1e-9 for x,y in zip(samples,timing['latency_samples_ms']))
   report['timings'][variant]={'measurement':timing,'offline_replay':str(a.windows)+'/'+str(a.windows)+' passed','timed_output_correctness':'passed'}
   (a.output/'progress.json').write_text(json.dumps(report,indent=2,default=str))
  report['status']='measured_pending_guard_and_environment_acceptance';report['phase']='finished'
 except Exception as e:
  report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc();raise
 finally:
  (a.output/'admission.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps(report,default=str),flush=True)
if __name__=='__main__':main()
