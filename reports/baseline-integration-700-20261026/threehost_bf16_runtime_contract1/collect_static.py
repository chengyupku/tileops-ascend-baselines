import pathlib,json,subprocess,hashlib
P=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/root/dyq/workspace')
files={'TileOPs/workloads/gemm.py':[(364,389)],'TileOPs/scripts/ascend_pair_check.py':[(137,162),(329,337)],'TileOPs/benchmarks/ascend_math_more.py':[(63,74)],'reports/baseline-integration-700-20261026/threehost_bmm1/binding.py':[(1,40)],'reports/baseline-integration-700-20261026/threehost_bmm1/server1/capture.py':[(27,50)],'.venv/lib/python3.12/site-packages/torch/version.py':[(1,12)],'.venv/lib/python3.12/site-packages/torch/include/ATen/Config.h':[(1,25)]}
def get(p,ranges):
 b=p.read_bytes();l=b.decode().splitlines();return {'sha256':hashlib.sha256(b).hexdigest(),'excerpts':[{'first':a,'text':'\n'.join(l[a-1:z])}for a,z in ranges]}
out={'local':{f:get(root/f,r) for f,r in files.items()}}
remote_files={'backends/mkldnn/__init__.py':[(17,25),(81,129)],'include/ATen/Context.h':[(473,486),(514,526)],'include/ATen/Config.h':[(1,25)],'_torch_docs.py':[(1526,1569)],'__init__.py':[(1591,1596),(1630,1648)],'include/oneapi/dnnl/dnnl.h':[(303,315),(364,383)],'include/oneapi/dnnl/dnnl_common_types.h':[(134,170)],'version.py':[(1,12)]}
code='from pathlib import Path\nimport json,hashlib\np=Path("/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch")\nfiles='+repr(remote_files)+'\no={}\nfor f,ranges in files.items():\n b=(p/f).read_bytes();l=b.decode().splitlines();o[f]={"sha256":hashlib.sha256(b).hexdigest(),"excerpts":[{"first":a,"text":"\\n".join(l[a-1:z])} for a,z in ranges]}\nprint(json.dumps(o))\n'
s=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-661495','python3 -S -'],input=code,text=True,capture_output=True,timeout=25);assert s.returncode==0,s.stderr
out['server1_installed_torch']=json.loads(s.stdout)
(P/'STATIC_EVIDENCE.json').write_text(json.dumps(out,indent=2)+'\n')
print('static evidence saved')
