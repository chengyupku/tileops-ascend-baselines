import sys,pathlib,os,json,time
P=pathlib.Path('/opt/torch-baseline1135-20261103');sys.path[:0]=[str(P/'source/src'),str(P/'source'),str(P),'/opt/tileops-ci/runtime/lib/python3.12/site-packages'];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
import workloads.attention.gqa.prefill_paged_kv_append as mod
T=json.loads((P/'server2/runs/8a5b6f2667856eac3bd2/a5/task.json').read_text())
torch.set_num_threads(8);set_run_device('cpu');torch.manual_seed(T['seed'])
c=next(c for c in cases(T['key'][1]) if c.id==T['key'][2]);xs=c.inputs
expected=c.reference(*xs);ys=tuple(x.npu() for x in xs)
original=mod.apply_dense_rope;tables=[]
def capture(x,index,cos,sin,**kw):
 tables.append((cos.cpu(),sin.cpu()));return original(x,index,cos,sin,**kw)
mod.apply_dense_rope=capture
got=c.reference(*ys);torch.npu.synchronize();got=tuple(x.cpu() for x in got);mod.apply_dense_rope=original
result={}
def check(label,gold):
 try:compare_outputs(got,gold,c.verification);result[label]={'pass':True}
 except AssertionError as e:result[label]={'pass':False,'error':str(e)}
 result[label]['outputs']=[{'shape':list(a.shape),'max_abs':float((a.float()-b.float()).abs().max()),'different':int((a!=b).sum())} for a,b in zip(got,gold)]
check('original_cpu',expected)
it=iter(tables)
def replay(x,index,cos,sin,**kw):
 a,b=next(it);return original(x,index,a,b,**kw)
mod.apply_dense_rope=replay
oracle=c.reference(*xs);mod.apply_dense_rope=original
check('cpu_actual_tables',oracle)
result['table_count']=len(tables);result['at']=time.time()
torch.save(tables,P/'server2/GQA_FREQUENCY_TABLES.pt')
(P/'server2/GQA_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));torch.npu.synchronize()
