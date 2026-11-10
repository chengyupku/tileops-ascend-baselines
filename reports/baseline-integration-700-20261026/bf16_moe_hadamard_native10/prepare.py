from pathlib import Path
import json,hashlib,shutil,importlib.util,sys
T=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace'); D=T.parent/'bf16_local_gemm18'; parent=T.parent/'bf16_moe_hadamard_parent10'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Fixed exact scope keys and manifest parameters copied from original manifests (read-only).
rows=[('MoEGroupedGemmFwdOp','qwen3-30b-ep8-decode-bfloat16'),('MoEGroupedGemmFwdOp','qwen3-30b-ep8-routed256-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-decode-down-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-decode-down-bfloat16-float32'),('MoEGroupedGemmFwdOp','ds-v3-ep8-masked-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-prefill-down-bfloat16'),('MoEGroupedGemmFwdOp','ds-v3-prefill-mlp-up-bfloat16'),('HadamardTransformFwdOp','llama-8b-prefill-bfloat16'),('HadamardTransformFwdOp','llama3-head-dim-bfloat16')]
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
from benchmarks.api import cases
spec=[]
for op,cid in rows:
 c=next(x for x in cases(op) if x.id==cid)
 p=dict(c.params)
 # derive full tensor envelope from manifest params without materializing device tensors
 if op=='HadamardTransformFwdOp':
  elems=1
  for d in p['x'][0]: elems*=int(d)
  inp=[list(p['x'][0])]
 else:
  ash=tuple(p['a'][0]); bsh=tuple(p['b'][0]); outsh=tuple(p['output'][0]); elems=sum(__import__('math').prod(z) for z in (ash,bsh,outsh)); inp=[list(ash),list(bsh),list(p['layout_metadata'][0])]
 spec.append({'key':['ops-transformer',op,cid],'manifest_params':p,'input_shapes':inp,'input_elements':elems,'envelope':{'host_required':2*1024**3+max(elems*64,1),'device_required':2*1024**3+max(elems*32,1),'disk_required':1024**3+max(elems*8,1)}})
(T/'CASES.json').write_text(json.dumps(spec,indent=2,default=str))
# Verify immutable release and parent adapter hashes.
release=json.loads((T.parent/'DEPLOYMENT_RELEASE_BF16_V5_20261026T2333.json').read_text())
for k in ('binding','native_host','device_object','original_generated_recipe'):
 assert sha(release[k]['path'])==release[k]['sha256'],k
pins={'release':release,'parent_adapter':sha(parent/'adapter.py'),'parent_cpu_checks':sha(parent/'CPU_CHECKS.json'),'orig_binding':sha(D/'binding.py'),'native_host':release['native_host']['sha256'],'device_object':release['device_object']['sha256'],'recipe':release['original_generated_recipe']['sha256']}
(T/'FROZEN.json').write_text(json.dumps(pins,indent=2))
(T/'PREPARATION.json').write_text(json.dumps({'rows':len(spec),'families':{'MoE':7,'Hadamard':2},'fixed_scope':True,'parent_adapter_reused':True,'numeric_validation':False,'frozen':pins},indent=2,default=str))
print('prepared',len(spec))
