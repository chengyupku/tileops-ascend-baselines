from pathlib import Path
import json,ast,hashlib
W=Path(__file__).resolve().parent;P=W/'ssd_package';P.mkdir(exist_ok=True);R=W.parents[2]
r=json.loads((W/'causal_package/RUNTIME_RECIPE.json').read_text());r['required_environment']['ASCEND_CUSTOM_OPP_PATH']='';(P/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2));(P/'direct_acl.py').write_text((W/'causal_package/direct_acl.py').read_text())
s='''from pathlib import Path
import json,hashlib
from direct_acl import require_guard
HERE=Path(__file__).resolve().parent
_LIB=None
def build(case):
 global _LIB
 require_guard()
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 assert case.id=='mamba2-780m-float16-float32'
 binary=HERE.parent/'ssd_compile_env_fixed/liboriginal_chunk_state.so'
 _LIB=torch.library.Library('npu_ops_transformer_ext','FRAGMENT')
 _LIB.define('mambav2_chunk_state(Tensor dt_out, Tensor dacs, Tensor bt, Tensor xt) -> Tensor')
 torch.ops.load_library(str(binary))
 def run(x,Bmat,dt,dA,seq_idx):
  assert seq_idx is None
  b,h,nc,q=dt.shape
  out=torch.ops.npu_ops_transformer_ext.mambav2_chunk_state(dt.permute(0,2,3,1).contiguous(),dA.permute(0,2,3,1).contiguous(),Bmat.reshape(b,nc,q,*Bmat.shape[2:]).contiguous(),x.reshape(b,nc,q,*x.shape[2:]).contiguous())
  return out.transpose(-1,-2).contiguous()
 rec=json.loads((HERE.parent/'ssd_compile_env_fixed/RECEIPT.json').read_text())
 hashes={str(binary):hashlib.sha256(binary.read_bytes()).hexdigest(),rec['source']:rec['source_sha256'],str(HERE/'binding.py'):hashlib.sha256((HERE/'binding.py').read_bytes()).hexdigest()}
 return Candidate(Implementation(run),'ops-transformer::mambav2_chunk_state','Original complete Torch API with shape/layout binding and full FP32 output transpose.',hashes,timing_mode='host_roundtrip')
''';(P/'binding.py').write_text(s);ast.parse(s)
s=(W/'causal_package/entry.py').read_text().replace("=='CausalConv1dUpdateFwdOp'","=='SSDChunkStateFwdOp'");(P/'entry.py').write_text(s)
i='mamba2-780m-float16-float32';d=W/i;d.mkdir(exist_ok=True)
s=(W/'mamba2-2.7b-decode-b1-bfloat16/child.sh').read_text().replace('mamba2-2.7b-decode-b1-bfloat16',i).replace('causal_package/entry.py','ssd_package/entry.py').replace('--op CausalConv1dUpdateFwdOp','--op SSDChunkStateFwdOp');pos=s.index('exec /opt');s=s[:pos]+'export ASCEND_CUSTOM_OPP_PATH=""\n'+s[pos:]
for line in s.splitlines():
 if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
(d/'child.sh').write_text(s)
print('SSD ready')
