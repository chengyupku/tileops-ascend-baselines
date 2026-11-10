import json,pathlib,subprocess,time,hashlib
p=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense');d=json.loads((p/'gelu_original_recipe.json').read_text())
replace={'/root/dyq/workspace/.tools/ascend/compiler/bin/bisheng':'/usr/local/Ascend/cann/aarch64-linux/bin/bisheng','/root/dyq/workspace/.venv':'/opt/tileops-ci/runtime','/root/dyq/workspace/.tools/ascend/cann':'/usr/local/Ascend/cann','/root/dyq/workspace/ascend_baselines/ops-transformer/experimental/moe/biasgategelu/BIAS_GATE_GELU.cpp':str(p/'BIAS_GATE_GELU.cpp'),'/root/dyq/workspace/.cache/all-pairs/ops_transformer/gelu_build':str(p),'/usr/include/python3.12':'/opt/tileops-ci/python/cpython-3.12-linux-aarch64-gnu/include/python3.12'}
def conv(x):
 for a,b in replace.items(): x=x.replace(a,b)
 return x.replace('/x86_64-linux/','/aarch64-linux/')
out={'source_sha256':hashlib.sha256((p/'BIAS_GATE_GELU.cpp').read_bytes()).hexdigest(),'NPU_used':False,'scope':'compile original single-file source with previous original recipe; paths relocated only'}
for phase in ['compile','link']:
 cmd=[conv(x) for x in d[phase]];t=time.monotonic()
 with (p/(phase+'.log')).open('w') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 out[phase]={'command':cmd,'returncode':rc,'elapsed_s':time.monotonic()-t}
 (p/'BUILD.json').write_text(json.dumps(out,indent=2))
 if rc:break
print(json.dumps(out,indent=2))
