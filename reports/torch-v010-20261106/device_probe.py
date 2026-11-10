import sys,os,pathlib,json,hashlib
sys.path.insert(0,'/opt/tileops-ci/runtime/lib/python3.12/site-packages');os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch_npu
torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
p=torch.npu.get_device_properties(0);torch.npu.synchronize();libs={}
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
 f=line.split()[-1]
 if f.startswith('/') and '.so' in f and any(x in f for x in ['Ascend','torch','libstdc++']):
  path=pathlib.Path(f)
  if path.is_file() and f not in libs:
   with path.open('rb') as fp:libs[f]=hashlib.file_digest(fp,'sha256').hexdigest()
print(json.dumps({'name':torch.npu.get_device_name(0),'cube':p.cube_core_num,'vector':p.vector_core_num,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'matmul_hf32':torch.npu.matmul.allow_hf32,'conv_hf32':torch.npu.conv.allow_hf32,'precision_mode':'must_keep_origin_dtype','loaded_libraries':libs},indent=2))
