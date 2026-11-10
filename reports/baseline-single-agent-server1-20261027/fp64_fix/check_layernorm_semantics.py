import os,sys,importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','aclnn','tilelang','triton'):raise ImportError('CPU-only analysis forbids '+fullname)
sys.meta_path.insert(0,BlockDevice())
from pathlib import Path
import torch,json,difflib,hashlib
P=Path(__file__).resolve().parent;W=P.parents[2];source=W/'TileOPs/workloads/norm.py'
s=source.read_text();needle='    fused = name in ("FusedAddRMSNormFwdOp", "FusedAddLayerNormFwdOp")\n'
branch='''    if name == "FusedAddLayerNormFwdOp":
        # Fused semantics: normalize the FP32 residual sum, round only outputs.
        summed = x.float() + inputs[1].float()
        weight, bias = inputs[2:]
        shape = tuple(p.get("normalized_shape", (x.shape[-1],)))
        output = F.layer_norm(summed, shape, weight.float(), bias.float(), eps)
        return output.to(x.dtype), summed.to(x.dtype)
'''
assert s.count(needle)==1
proposal=s.replace(needle,branch+needle)
(P/'add_layernorm_reference.PROPOSED.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),proposal.splitlines(True),fromfile='a/TileOPs/workloads/norm.py',tofile='b/TileOPs/workloads/norm.py')))
torch.set_num_threads(4); results=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 torch.manual_seed(130)
 x=torch.randn(256,4096,dtype=dtype);r=torch.randn_like(x);w=torch.randn(4096,dtype=dtype);b=torch.randn_like(w)
 summed=x.float()+r.float();stored=summed.to(dtype)
 old=torch.nn.functional.layer_norm(stored.float(),(4096,),w.float(),b.float(),1e-5).to(dtype)
 fused=torch.nn.functional.layer_norm(summed,(4096,),w.float(),b.float(),1e-5).to(dtype)
 err=(old.float()-fused.float()).abs()
 results.append({'dtype':str(dtype),'differing_elements':int((old!=fused).sum()),'max_abs_difference':float(err.max()),'same_residual_output':True})
assert results[0]['differing_elements']>0 and results[-1]['differing_elements']==0
(P/'ADD_LAYERNORM_SEMANTICS.json').write_text(json.dumps({'cpu_only':True,'device_imports_blocked':True,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(s.encode()).hexdigest(),'proposal_applied':False,'candidate_run':False,'manifest_acceptance':False,'purpose':'Synthetic counterexample establishes these are distinct precision contracts, not a layout-only repair. Not canonical NPU inputs or case verification.','results':results},indent=2)+'\n')
print(json.dumps(results))
