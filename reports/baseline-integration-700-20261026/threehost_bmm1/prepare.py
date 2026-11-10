from pathlib import Path
import subprocess,json,hashlib,shlex,time
P=Path(__file__).resolve().parent; BASE=P.parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOST='openlibing-DevEnv-661495'; R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1'
def remote(s):
 r=subprocess.run([SSH,HOST,s],capture_output=True,text=True); assert r.returncode==0,(r.returncode,r.stdout,r.stderr);return r.stdout
for name in ['adapter.py','capture.py','native_observation.py','resource_gate.py']:
 s=(BASE/'bf16_reference_budget2'/name).read_text()
 if name=='adapter.py':s=s.replace("task.parent/'bf16_local_gemm18/binding.py'","task/'binding.py'")
 if name=='capture.py':
  s=s.replace("W=Path('/root/dyq/workspace')",f"W=Path('{R}/frozen_workspace')")
  s=s.replace("sys.path[:0]=", "sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')\nsys.path[:0]=")
  s=s.replace("if '/bf16gmm_transformer/' in p", "if '/private_provider/' in p")
 (P/name).write_text(s)
s=(BASE/'bf16_local_gemm18/binding.py').read_text().replace("P/'install/vendors/bf16gmm_transformer'","P/'private_provider'")
(P/'binding.py').write_text(s)
rows=json.loads((BASE/'bf16_reference_budget2/CASES.json').read_text());(P/'CASES.json').write_text(json.dumps([r for r in rows if r['key'][1]=='BmmFwdOp'],indent=2))
setup=f'''import pathlib,shutil,json,hashlib,struct,os
p=pathlib.Path({R!r});p.mkdir(parents=True,exist_ok=True)
w=pathlib.Path('/opt/tileops-all-pairs/workspace');f=p/'frozen_workspace';f.mkdir(exist_ok=True)
for name in ['benchmarks','workloads','src','scripts']:
 shutil.copytree(w/'TileOPs'/name,f/'TileOPs'/name,ignore=shutil.ignore_patterns('__pycache__','.git'),dirs_exist_ok=True)
shutil.copytree(w/'tools',f/'tools',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
v=w/'reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider'
shutil.copytree(v,p/'private_provider',dirs_exist_ok=True)
b=p/'private_provider/op_api/lib/libcust_opapi.so';raw=b.read_bytes();assert int.from_bytes(raw[18:20],'little')==183
meta={{'machine':'aarch64','elf_machine':183,'binary_size':len(raw),'binary_sha256':hashlib.sha256(raw).hexdigest(),'original_provider':str(v),'snapshot':str(p/'private_provider'),'disk_free':shutil.disk_usage(p).free}}
(p/'ABI.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
'''
(P/'remote_setup.py').write_text(setup)
print(remote('python3 -c '+shlex.quote(setup)))
# Transfer only Python/text files, never any x86 native artifact.
for name in ['adapter.py','capture.py','native_observation.py','resource_gate.py','binding.py','CASES.json']:
 subprocess.run([SSH,HOST,'cat > '+shlex.quote(R+'/'+name)],input=(P/name).read_bytes(),check=True)
wrapper=remote('cat /opt/tileops-all-pairs/bin/allpairs-python')
assert '\\n' not in wrapper, 'literal escape in runtime wrapper'
wrapper=wrapper[:wrapper.index('exec /opt/tileops-ci/runtime/bin/python')]
run=wrapper+f'''\ncd {R}/frozen_workspace
export ASCEND_CUSTOM_OPP_PATH={R}/private_provider
export ASCEND_CLASSIC_TRANSFORMER_VENDOR=$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_GLOBAL_LOG_LEVEL=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
for name in ASCEND_CACHE_PATH ASCEND_WORK_PATH XDG_CACHE_HOME TORCH_EXTENSIONS_DIR ASCEND_PROCESS_LOG_PATH TRITON_CACHE_DIR TMPDIR TMP TEMP ACL_OP_COMPILER_CACHE_DIR; do
 export "$name={R}/square-4k-bfloat16/$name"
 mkdir -p "${{!name}}"
done
exec /opt/tileops-ci/runtime/bin/python -I -S -B /opt/tileops-all-pairs/control/private_python.py {R}/capture.py {R}/square-4k-bfloat16 BmmFwdOp square-4k-bfloat16
'''
(P/'run.sh').write_text(run)
subprocess.run([SSH,HOST,'cat > '+shlex.quote(R+'/run.sh')],input=run.encode(),check=True)
freeze=f'''from pathlib import Path
import hashlib,json,ast
p=Path({R!r});pins={{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in p.rglob('*') if x.is_file()}}
for x in p.glob('*.py'):ast.parse(x.read_text())
(p/'FROZEN.json').write_text(json.dumps({{'sha256':pins}},indent=2));print(len(pins))
'''
print(remote('python3 -c '+shlex.quote(freeze)))
print(remote('/opt/tileops-ci/runtime/bin/python -I -S -B /opt/tileops-all-pairs/control/private_python.py -c '+shlex.quote("import sys; print(sys.version); print('CPU runtime entry OK')")))
(P/'CPU_PREPARE.json').write_text(json.dumps({'time':time.time(),'status':'deployed_private_snapshot','host':'server1','runtime_wrapper_literal_backslash_n':False},indent=2))
