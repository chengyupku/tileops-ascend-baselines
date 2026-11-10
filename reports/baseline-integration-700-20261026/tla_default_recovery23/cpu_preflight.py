import ast, pathlib, json, hashlib, shutil, subprocess, os, datetime, sys, importlib.machinery
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/tla_default_recovery23';P.mkdir(parents=True,exist_ok=True)
R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
V=W/'ascend_baselines/tilelang-ascend';old=W/'reports/baseline-integration-700-20261026/tla_simple_tilelang';sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
E=P/'cpu_attempt01';E.mkdir(exist_ok=False)
records=[]
def freeze(f,name):
 d=E/name;shutil.copyfile(f,d);records.append(dict(original=str(f),frozen=str(d),sha256=hashlib.sha256(d.read_bytes()).hexdigest()));return d
lib=freeze(R/'tilelang/jit/adapter/libgen.py','runtime_libgen.py')
freeze(V/'tilelang/jit/adapter/libgen.py','vendor_libgen.py')
freeze(R/'tilelang/transform/pass_config.py','pass_config.py')
freeze(R/'tilelang/jit/__init__.py','jit_init.py')
freeze(R/'tilelang/engine/lower.py','lower.py')
for rel in ['examples/batch_gemm/batch_gemm.py','examples/gemm/example_gemm.py']:
 freeze(V/rel,pathlib.Path(rel).name)
for name in ['gemm_adapter.py','run_batch5.py','gemm_probe.guard.json','gemm_probe.guard.log']:
 freeze(old/name,name)
case=old/'runs/BmmFwdOp_square-256-bfloat16_v3'
freeze(case/'checker.json','prior_checker.json')
for f in (case/'jit').rglob('*'):
 if f.is_file():freeze(f,f.name)
src=freeze(old/'cache/TMPDIR/tmpfia3pciv.cpp','original_auto.cpp')
# Evaluate only the unchanged literal command-list expression; never import compiler/runtime.
tree=ast.parse(lib.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="self.target == 'ascendc' or self.target == 'auto'")
assignment=next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='command' for t in n.targets))
class Source: name=str(src)
env=os.environ.copy();env.update(ASCEND_HOME_PATH=str(sdk),ASCEND_HOME=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
cmd=eval(compile(ast.Expression(assignment.value),str(lib),'eval'),{'ASCEND_HOME_PATH':str(sdk),'TL_ROOT':str(V),'TILELANG_TEMPLATE_PATH':str(R/'src'),'src':Source()})
# SDK selection only; driver dry run performs no compilation, linking, loading or launch.
cmd[0]=str(sdk/'tools/bisheng_compiler/bin/bisheng');cmd += ['-###','-o',str(E/'unused.so')]
r=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=E,timeout=30)
(E/'sdk92_driver.log').write_text(r.stdout+r.stderr)
(E/'sdk92_driver.json').write_text(json.dumps(dict(argv=cmd,returncode=r.returncode,device_access=False,mode='driver dry run; original auto command list plus -### and private output'),indent=2))
paths=[W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages',R,V/'3rdparty/tvm/python',W/'.cache/all-pairs/remote_tla_compiler/python',W/'.cache/all-pairs/remote_native_tla/python-deps',pathlib.Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages')]
deps={}
for name in ['torch','torch_npu','tilelang','tvm','numpy','setuptools','Cython']:
 spec=importlib.machinery.PathFinder.find_spec(name,[str(p) for p in paths]);deps[name]=None if spec is None else spec.origin
(E/'dependencies.json').write_text(json.dumps(dict(python=sys.executable,version=sys.version,modules_not_imported=True,paths=[str(p) for p in paths],dependencies=deps),indent=2))
(E/'hashes.json').write_text(json.dumps(records,indent=2))
print(json.dumps(dict(dependencies=deps,driver_rc=r.returncode,driver_log=str(E/'sdk92_driver.log'),frozen_count=len(records)),indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='cpu_driver_completed',npu_attempts=0),indent=2))
