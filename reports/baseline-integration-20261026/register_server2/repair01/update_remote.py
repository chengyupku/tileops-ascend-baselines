import subprocess,io,tarfile
from pathlib import Path
D=Path(__file__).resolve().parent;ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409']
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for n in ['validated_runtime.py','launch_verified.py']:t.add(D/n,arcname=n)
subprocess.run(ssh+['tar -xmf - -C /opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer'],input=b.getvalue(),check=True)
s="""
from pathlib import Path
import json
p=Path('/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json');m=json.loads(p.read_text());c=m['groups']['attention']
c['ld_library_path_prefix']='/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib:/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib'
c['directories']+=c['ld_library_path_prefix'].split(':')
c['environment'].update(ASCEND_AICPU_PATH='/usr/local/Ascend/cann-9.1.0',TRITON_BACKEND='torch_npu',ATB_HOME_PATH='/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1')
p.write_text(json.dumps(m,indent=2)+'\\n')
"""
subprocess.run(ssh+['python3 -S -B -'],input=s.encode(),check=True)
