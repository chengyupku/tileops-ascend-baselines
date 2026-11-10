import sys,pathlib,json,os,importlib.util,hashlib
P=pathlib.Path(__file__).resolve().parent
import torch,torch_npu
torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.set_num_threads(8)
spec=importlib.util.spec_from_file_location('parser','/opt/torch-all2208-20261105/tools/kernel_task_profile_full32.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
x=torch.tensor([0,1],dtype=torch.int16).repeat(8388608).to('npu').reshape(262144,64)
def int32eq():
 y=x.to(torch.int32);return (y==y).to(torch.int16)
def bool_or():
 y=x.bool();return (y | torch.logical_not(y)).to(torch.int16)
def float_ones():return torch.ones(x.shape,dtype=torch.float32,device=x.device).to(torch.int16)
def int32_ones():return torch.ones(x.shape,dtype=torch.int32,device=x.device).to(torch.int16)
rows=[]
for name,fn in [('int32eq',int32eq),('bool_or',bool_or),('float_ones',float_ones),('int32_ones',int32_ones)]:
 fn();torch.npu.synchronize();os.environ['ASCEND_WORK_PATH']=str(P/('constant-probe-'+name))
 with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=lambda _:None,experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,l2_cache=False,data_simplification=False)) as prof:
  got=fn();torch.npu.synchronize()
 cpu=got.cpu();assert torch.equal(cpu,torch.ones_like(cpu))
 path=pathlib.Path(prof.prof_if.prof_path);r=parser.parse_profile(path,1);rows.append({'name':name,'exact':True,'profile_path':str(path),'trace_sha256':parser.trace_digest(path),**r});print(name,r['latency_ms'],r['profile_aicpu_task_count'],flush=True)
(P/'CONSTANT_PROBES.json').write_text(json.dumps({'scope':'diagnostic no flush; not release timings','torch':torch.__version__,'torch_npu':torch_npu.__version__,'torch_npu_file':torch_npu.__file__,'rows':rows},indent=2));torch.npu.synchronize()
