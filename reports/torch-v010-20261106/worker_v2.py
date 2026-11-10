"""Frozen CPU preparation and guarded NPU measurement; no task starts at import."""
import os,sys,pathlib,json,time,hashlib,traceback,dataclasses,importlib.abc
P=pathlib.Path(__file__).resolve().parent
mode,taskfile,outdir=sys.argv[1:];task=json.loads(pathlib.Path(taskfile).read_text());out=pathlib.Path(outdir);out.mkdir(parents=True,exist_ok=True)
source_root=P/('source_950' if task.get('source_contract')=='950_extension' else 'source')
sys.path[:0]=[str(source_root/'src'),str(source_root),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ['FLA_NPU_DISABLE_PTH']='1'
def put(name,data):
 f=out/name;q=f.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str));q.replace(f)
def sha(f):
 with pathlib.Path(f).open('rb') as fp:return hashlib.file_digest(fp,'sha256').hexdigest()
def source_proof():
 d=json.loads((P/'SOURCES.v2.json').read_text());assert all(sha(P/n)==h for n,h in d.items()),'frozen source changed';return sha(P/'SOURCES.v2.json')
if mode=='prepare':
 class NoNPU(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in ('torch_npu','acl'):
    raise ModuleNotFoundError('CPU prepare forbids '+fullname,name=fullname)
 sys.meta_path.insert(0,NoNPU())
elif mode=='device':
 import fcntl
 with open('/run/lock/ascend-npu/device-0.lock') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('device requires held host guard')
else:raise ValueError(mode)
started=time.time()
try:
 proof=source_proof()
 import torch
 torch.set_num_threads(32 if mode=='prepare' and task['key'][1]=='DSADecodeWithKVCacheFwdOp' else 8)
 from benchmarks.api import cases
 from workloads.device import set_run_device
 from workloads.numerics import compare_outputs,describe
 from tools import kernel_task_profile_full32 as full32
 sys.modules['tools.kernel_task_profile']=full32
 from tools import measure_baselines as core
 core.ROOT=P;core.TILEOPS_ROOT=source_root
 set_run_device('cpu');torch.manual_seed(task['seed'])
 if task['key'][1]=='DeltaNetFwdOp':
  import workloads.linear_attention.deltanet as delta
  from adapters.delta_reference import reference as delta_reference
  def delta_route(*a,scale=None,l2norm=False):return delta_reference(*a,scale=scale,use_qk_l2norm_in_kernel=l2norm)
  delta.deltanet_inference_ref=delta_route
  put('reference_route.json',{'formula':'direct Delta recurrence; CPU FP64, device FP32','replaces':'FLA-only entry; current generator/checker unchanged','adapter':'delta_reference'})
 if task['key'][1]=='GLAFwdOp':
  import workloads.linear_attention.gla as gla
  from adapters import gla_reference,gla_eager
  gla.gla_inference_ref=gla_reference.reference if mode=='prepare' else gla_eager.reference
  put('reference_route.json',{'formula':'same GLA recurrence; CPU FP64/device chunked FP32','adapter':'gla_reference/gla_eager','current_generator_checker_unchanged':True})
 if task['id']=='87bc414b0c6ff0eaee1f':
  import workloads.mamba as mamba
  from adapters.ssd_chunk_memory_v17 import bind
  bind(mamba)
  put('ssd_memory_route.json',{'independent_chunks_per_call':4,'original_per_chunk_function':True,'original_reduction_order':True,'scope':'large SSDChunkState memory bound','CPU_and_device_same_arithmetic':True})
 oracle_label=('independent pure torch recurrence; current FLA entry replaced' if task['key'][1] in ('DeltaNetFwdOp','GLAFwdOp') else ('independent CPU six-output triangular solve; final output crosschecked against canonical reference' if task['key'][1]=='DeltaNetChunkFwdOp' else 'canonical reference independently executed on CPU'))
 case=next(c for c in cases(task['key'][1]) if c.id==task['key'][2])
 if task['key'][1]=='GLAChunkFwdOp':
  from workloads.linear_attention.gla import GLAChunkwiseCall
  case.__dict__['workload']=GLAChunkwiseCall(case._call)
  put('checker_route.json',{'reason':'full torch output/state checker; FLA-specific Partial checker not used','source':'GLAChunkwiseCall.verification'})
 assert core._jsonable(case.params)==core._jsonable(task['case']['params']),'exact-case parameters changed'
 def convert(x,device):
  if isinstance(x,torch.Tensor):return x.to(device)
  if isinstance(x,tuple):return tuple(convert(v,device) for v in x)
  if isinstance(x,list):return [convert(v,device) for v in x]
  if isinstance(x,dict):return {k:convert(v,device) for k,v in x.items()}
  return x
 if mode=='prepare':
  if task['key'][1]=='FusedMoESharedExpertFwdOp':
   # CPU-only memory admission is serialized through prepare process exit.
   import fcntl
   moe_prepare_lock=(P/'moe-prepare-memory.lock').open('a')
   fcntl.flock(moe_prepare_lock,fcntl.LOCK_EX)
   import math
   dtype_sizes={'float32':4,'bfloat16':2,'float16':2,'int64':8,'int32':4,'int8':1,'uint8':1}
   input_bytes=sum(math.prod(v[0])*dtype_sizes.get(v[1],4) for v in task['case']['tensors'].values() if isinstance(v,list) and len(v)==2 and isinstance(v[0],list))
   cg=pathlib.Path('/sys/fs/cgroup/memory');deadline=time.monotonic()+1800
   while cg.exists():
    limit=int((cg/'memory.limit_in_bytes').read_text());usage=int((cg/'memory.usage_in_bytes').read_text())
    stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines());inactive=int(stat.get('total_inactive_file',stat.get('inactive_file','0')))
    working=max(0,usage-inactive)
    # Large host payload can be loaded by the NPU worker after this CPU sample.
    # For >28GiB inputs, wait for prior prepared attempts to close, avoiding that race.
    prior_payloads=[]
    if input_bytes>28*1024**3:
     prior_payloads=[f for f in P.glob('server*/runs/*/a*/PREPARED.json') if f.parent!=out and not (f.parent/'DONE.json').exists() and not (f.parent/'CANCELLED.json').exists()]
    if not prior_payloads and working+2.2*input_bytes+8*1024**3<limit:break
    put('CPU_MEMORY_WAIT.json',{'at':time.time(),'working_bytes':working,'expected_input_bytes':input_bytes,'limit':limit})
    if time.monotonic()>deadline:raise RuntimeError('CPU memory admission unavailable for large MoE preparation')
    time.sleep(2)
  if task['key'][1]=='DeltaNetChunkBwdOp':
   from adapters.delta_six_cpu_v2 import reference as cpu_six
   do,q,k,v,beta,*unused=case.workload.gen_inputs()
   _,state,aw,au,w,u=cpu_six(q,k,v,beta,case._call.params['chunk_size'])
   case.__dict__['inputs']=(do,q,k,v,beta,state,aw,au,w,u)
   put('input_preparation_route.json',{'original_generator':True,'saved_tensors':'independent CPU forward, same current inputs','reason':'canonical input factory invokes device-only TileOPs forward on CPU'})
  if task['key'][1]=='GLAChunkBwdOp':
   from workloads.linear_attention.gla import gla_fwd_chunked_torch
   do,old_dht,q,k,v,g,old_h=case.workload.gen_inputs()
   params=case.workload.arguments()
   initial=torch.randn_like(old_dht)*0.1 if params['has_initial_state'] else None
   _,states=gla_fwd_chunked_torch(q,k,v,g,params['chunk_size'],scale=params['scale'],initial_state=initial,with_chunk_states=True)
   case.__dict__['inputs']=(do,torch.zeros_like(old_dht),q,k,v,g,states.float())
   put('input_preparation_route.json',{'original_generator':True,'saved_tensors':'v0.1.0 CPU forward chunk states, same initial-state policy, zero final gradient'})
  xs=case.inputs
  assert case.reference is not None,'missing reference: implement route before enqueue'
  # A CPU reference may declare in-place outputs; preserve the original sample.
  def clone(x):
   if isinstance(x,torch.Tensor):return x.clone()
   if isinstance(x,tuple):return tuple(clone(v) for v in x)
   return x
  expected=case.reference(*clone(xs))
  if task['key'][1] in ('GemmFwdOp','BmmFwdOp','Conv1dFwdOp','Conv2dFwdOp','GroupedGemmFwdOp'):
   promoted=tuple(x.double() if isinstance(x,torch.Tensor) and x.is_floating_point() else x for x in xs)
   high=case.reference(*promoted).to(expected.dtype)
   put('cpu_oracle_precision.json',{'policy':'PRECISION_POLICY.json','method':'linear contraction FP64 accumulation then original output dtype; no intermediate quantization removed'})
   expected=high
   del high,promoted
  put('prepare.json',{'cpu_only':True,'key':task['key'],'params':core._jsonable(case.params),'seed':task['seed'],'sources_sha256':proof,'cpu_seconds':time.time()-started,'oracle':oracle_label,'verification':describe(case.verification)})
  if task['key'][1]=='DeltaNetChunkFwdOp':
   from adapters.delta_six_cpu_v2 import reference as cpu_six
   six=cpu_six(*xs,case._call.params['chunk_size'])
   agreement=compare_outputs(six[0],expected,case.verification)
   put('cpu_six_agreement.json',dataclasses.asdict(agreement))
   expected=six
  if task['key'][1]=='GemmW4A16FwdOp':
   from workloads.gemm import dequantize_w4a16_weight
   weight=dequantize_w4a16_weight(*xs[1:]).to(xs[0].dtype)
   expected=(xs[0].double()@weight.double().T).to(xs[0].dtype)
   oracle_label='independent FP64 contraction after original logical-weight and activation-dtype rounding'
   put('w4a16_oracle.json',{'contraction':'FP64 CPU','weight_rounding':'original activation dtype before contraction','original_checker':True})
  auxiliary={}
  if task['key'][1]=='LongRoPEFwdOp' and case.workload.tensors.get('rescale_factors') is not None:
   auxiliary['rescale_factors']=case.workload.tensors['rescale_factors']
   put('auxiliary_input.json',{'source':'current call.materialize CPU input, not a derived frequency table','name':'rescale_factors','dtype':str(auxiliary['rescale_factors'].dtype),'shape':list(auxiliary['rescale_factors'].shape),'sha256':hashlib.sha256(auxiliary['rescale_factors'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
  bundle={'inputs':xs,'expected':expected,'auxiliary':auxiliary}
  from adapters.compressed_handoff_v010 import save
  put('payload.json',save(bundle,P,task['id'],timeout=600))
  assert not any(n=='torch_npu' or n.startswith('torch_npu.') for n in sys.modules)
  put('PREPARED.json',{'at':time.time(),'seconds':time.time()-started})
 else:
  import torch_npu
  torch.npu.set_device(0);set_run_device('npu')
  torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'})
  torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  put('precision_options.json',{'mode':'must_keep_origin_dtype','matmul_hf32':False,'conv_hf32':False,'policy':'PRECISION_POLICY.json'})
  props=torch.npu.get_device_properties(0);hardware={'device':torch.npu.get_device_name(0),'cube':props.cube_core_num,'vector':props.vector_core_num,'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__}
  assert (hardware['cube'],hardware['vector'])==(32,64),hardware
  payload=json.loads((out/'payload.json').read_text())
  from adapters.compressed_handoff_v010 import load
  saved=load(payload)
  cpu=saved['inputs'];expected=saved['expected'];case.__dict__['inputs']=cpu
  args=convert(cpu,'npu');fn=case.reference;reset=None;cleanup=None
  auxiliary=saved.get('auxiliary',{})
  if auxiliary:
   assert task['key'][1]=='LongRoPEFwdOp'
   case.workload.tensors.update(convert(auxiliary,'npu'))
   put('auxiliary_device_route.json',{'transfer':'same CPU-materialized rescale_factors bytes to device; no internally computed frequency table substituted','same_formula':True,'generation_outside_timing':True})
  if task['key'][1] in ('GDNFwdOp','KDAFwdOp'):
   from adapters import gated_fp32_v7 as route
   if task['key'][1]=='GDNFwdOp':
    import workloads.linear_attention.gdn as module
    module.gdn_ref=route.gdn_ref
   else:
    import workloads.linear_attention.kda as module
    module.kda_ref=route.kda_ref
   put('device_route.json',{'adapter':'gated_fp32_v7','oracle':'current canonical CPU unchanged','execution_precision':'FP32','requires_current_checker':True})
  if task['key'][1]=='GemmW4A8FwdOp':
   from adapters.w4a8_eager import gemm_w4a8
   def fn(*a):return gemm_w4a8(*a,group_size=case._call.params['group_size'])
   put('device_route.json',{'adapter':'adapters/w4a8_eager.py','formula':'signed INT4, FP8-rounded weight, FP32 contraction, token then channel scale, BF16 output','device_arithmetic':True,'original_cpu_checker':True,'all_operations_timed':True})
  if task['key'][1]=='FFTC2CFwdOp' and args[0].dtype==torch.complex128:
   from adapters.fft_ops_math_v5 import bind
   fn,cleanup=bind(tuple(args[0].shape))
   put('device_route.json',{'adapter':'adapters/fft_ops_math.py','origin':'ops-math original aclStft host API, explicit complex unity window','execution':'device AiCPU; not host CPU','device_arithmetic':True,'original_cpu_checker':True})
  if task['key'][1]=='GemmINT8W8A8FwdOp':
   from adapters.int8_gemm_v5 import reference as int8_reference
   def fn(*a):return int8_reference(case._call,*a)
   put('device_route.json',{'adapter':'adapters/int8_gemm_v5.py','integer_partial_bound':8388608,'device_arithmetic':True,'original_cpu_checker':True})
  if task['key'][1]=='DeltaNetChunkFwdOp':
   from adapters.delta_chunk_six_server1_v1 import reference as device_six
   def fn(*a):return device_six(*a,case._call.params['chunk_size'])
   put('device_route.json',{'adapter':'delta_chunk_six_server1_v1','CPU_oracle':'independent FP64 triangular solve','outputs':6,'device_arithmetic':True})
  if task['key'][1]=='DeltaNetChunkBwdOp':
   from adapters.delta_training_bwd_eager import reference as bwd
   def fn(*a):return bwd(*a,chunk_size=case._call.params['chunk_size'])
   put('device_route.json',{'adapter':'delta_training_bwd_eager','original_cpu_checker':True,'device_arithmetic':True})
  if task['key'][1]=='GemmW4A16FwdOp':
   from workloads.gemm import dequantize_w4a16_weight
   def fn(*a):
    weight=dequantize_w4a16_weight(*a[1:]).to(a[0].dtype)
    return (a[0].float()@weight.float().T).to(a[0].dtype)
   put('w4a16_route.json',{'contraction':'FP32 device','weight_rounding':'original activation dtype','original_checker':True,'all_operations_timed':True})
  if task['id']=='5f035f7b352732614242':
   from adapters.scan_precise_v16 import selective_scan_ref
   def fn(*a):return selective_scan_ref(*a,delta_softplus=case._call.params.get('delta_softplus',True))
   put('selective_scan_route.json',{'scope':'mamba-130m-no-gate-float32','method':'same FP32 recurrence; device native FP64 softplus intermediates and batched native PowAiCpu decay rounded FP32','oracle':'original canonical CPU unchanged','original_checker':True,'all_operations_timed':True})
  original_device_args=args
  original_inputs_before=core._tensor_digest(original_device_args)
  torch.npu.synchronize();boundary_started=time.perf_counter()
  from adapters.boundary_routes_v2 import bind as bind_boundary
  boundary=bind_boundary(case,args,fn,task.get('reasons',[]))
  fn,args,reset=boundary.execute,boundary.args,boundary.reset
  torch.npu.synchronize()
  boundary.description['setup_host_wall_seconds']=time.perf_counter()-boundary_started
  boundary.description['setup_cost_scope']='boundary construction only; excludes runtime initialization/input upload; synchronized host wall time, not device latency'
  put('timing_boundary.json',boundary.description)
  reset();torch.npu.synchronize()
  if any(isinstance(x,torch.Tensor) and x.dtype in (torch.float8_e4m3fn,torch.float8_e5m2) for x in args):
   from adapters.fp8_index_v4 import FP8Index
   fp8_fn=fn
   def fn(*a):
    with FP8Index():return fp8_fn(*a)
   put('fp8_index_route.json',{'method':'uint8 storage views; exact bytes gathered/scattered then FP8 view','no_numerical_conversion':True,'reject_accumulate':True,'inside_timed_call':True})
  if task['key'][1]=='GQAPrefillPagedWithKVCacheFwdOp':
   from adapters.fp8_saturating_v9 import FiniteFP8Saturation
   saturation_fn=fn
   def fn(*a):
    with FiniteFP8Saturation():return saturation_fn(*a)
   put('fp8_cast_route.json',{'basis':'CPU current torch finite-overflow saturates; NPU native cast returns NaN beyond464','finite_clamp':448,'inside_timed_call':True,'original_checker':True})
  if task['key'][1] in ('RoPEFwdOp','RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp','GQAPrefillPagedWithKVCacheFwdOp'):
   from adapters.rope_pow_v10 import RopePower
   rope_fn=fn
   def fn(*a):
    with RopePower():return rope_fn(*a)
   put('rope_precision_route.json',{'method':'native device PowAiCpu with FP64 base, exponent FP32, result rounded FP32','host_cpu_fallback':False,'original_checker':True,'all_operations_timed':True,'proof':'guarded raw hash identifies aclnnPowTensorTensor_PowAiCpu_Pow'})
  if task['key'][1]=='InstanceNormFwdOp' and args[0].dtype==torch.float16 and args[1] is None and args[2] is None:
   from adapters.instance_norm_v12 import reference as instance_reference
   def fn(*a):return instance_reference(*a,eps=case._call.params.get('eps',1e-5))
   put('instance_norm_route.json',{'precision':'FP32 reductions; saved mean/invstd rounded FP16 as CPU native_batch_norm; folded affine; original output dtype','original_cpu_reference_checker':True,'all_operations_timed':True,'scope':'FP16 input stats, no running stats'})
  assert original_inputs_before==core._tensor_digest(original_device_args),'boundary setup mutated original inputs'
  put('original_input_metadata.json',{'tensors':[{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),'offset':x.storage_offset(),'device':str(x.device)} if isinstance(x,torch.Tensor) else {'type':type(x).__name__} for x in original_device_args],'preservation_check':'prefix-byte-sample; includes original inputs even when execution uses a closed saved graph'})
  put('input_metadata.json',{'seed':task['seed'],'tensors':[{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),'offset':x.storage_offset(),'device':str(x.device)} if isinstance(x,torch.Tensor) else {'type':type(x).__name__} for x in args],'preservation_check':'prefix-byte-sample; not exhaustive, user deferred issue1'})
  before=core._tensor_digest(args)
  reset();torch.npu.synchronize()
  produced=fn(*args);torch.npu.synchronize()
  def device_outputs(x):
   if isinstance(x,torch.Tensor):assert x.device.type=='npu',str(x.device)
   elif isinstance(x,(tuple,list)):
    for v in x:device_outputs(v)
   elif isinstance(x,dict):
    for v in x.values():device_outputs(v)
  device_outputs(produced)
  policy=case.verification
  if task['key'][1]=='DeltaNetChunkFwdOp':
   from workloads.numerics import Exact
   original=compare_outputs(convert(produced,'cpu'),expected,policy)
   put('original_checker.json',dataclasses.asdict(original))
   policy=Exact(atol=policy.atol,rtol=policy.rtol)
  if task['id'] in ('4b8214a1b63b22c27078','5793002de1d3f13ff709'):
   check=compare_outputs(produced,convert(expected,'npu'),policy)
   put('checker_execution.json',{'source':'original unchanged Custom validator','execution_device':'npu','CPU_reference_values':'byte-preserving transfer; unchanged','thresholds':'unchanged','reason':'CPU float32 cosine self-similarity0.98699; same original checker on NPU self and actual both passed','outside_timing':True})
  else:
   check=compare_outputs(convert(produced,'cpu'),expected,policy)
  assert check.checked_outputs and not check.unchecked_reason,dataclasses.asdict(check)
  if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
  torch.npu.synchronize()
  reset();torch.npu.synchronize()
  assert before==core._tensor_digest(args),'inputs not restored after correctness'
  assert original_inputs_before==core._tensor_digest(original_device_args),'original inputs mutated during correctness'
  put('correctness.json',{'status':'verified','check':dataclasses.asdict(check),'oracle':oracle_label,'inputs_preserved':True,'policy':describe(case.verification),'source_sha256':proof})
  del produced,expected,saved,cpu
  # The already exported original 1GiB flush kernel; no compiler/schedule changes.
  import ctypes
  fd=P/'flush';src=json.loads((fd/'SOURCE.json').read_text());assert sha(fd/'host_kernel.asc')==src['generated_source_sha256'];assert sha(fd/'flush.aibin')==src['device_binary_sha256']
  dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
  assert dll.init(str(fd/'flush.aibin').encode())==0
  def flush(buffer):
   if dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('flush launch failed')
  buffer=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buffer);torch.npu.synchronize();assert torch.count_nonzero(buffer).item()==0
  core._flush_kernel_cache[0]=(flush,buffer)
  os.environ['TILEOPS_PROFILE_DIR']=str(out/'profiles')
  put('loaded_environment.json',{'expected_environment':json.loads((P/'ENVIRONMENT_EQUIVALENCE.json').read_text()),'precision':json.loads((out/'precision_options.json').read_text())})
  timing=core._time_reference(torch,torch_npu,fn,args,3,15,reset=reset)
  if reset:reset();torch.npu.synchronize()
  assert before==core._tensor_digest(args),'inputs modified during timing'
  assert original_inputs_before==core._tensor_digest(original_device_args),'original inputs mutated during timing'
  assert proof==source_proof()
  put('measurement.json',{'status':'measured_pending_guard','key':task['key'],'task_id':task['id'],'hardware':hardware,'timing':timing,'correctness_sha256':sha(out/'correctness.json'),'source_sha256':proof,'started_at':started,'finished_at':time.time(),'warmup':3,'repeats':15,'inputs_preserved':True})
  torch.npu.synchronize()
  if cleanup:cleanup()
except BaseException as e:
 put(mode+'_failure.json',{'at':time.time(),'key':task['key'],'error':str(e),'traceback':traceback.format_exc()});raise
