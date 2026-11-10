import ast,builtins,json,os,pathlib,subprocess,sys,types
O=pathlib.Path(__file__).resolve().parent
subprocess.run(['bash','-n',str(O/'run_environment.sh')],check=True)
raw=(O/'integration700_fp16_bmm_registration1.py').read_text();constants={}
for node in ast.parse(raw).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except (ValueError,TypeError):pass
root=O/'RECIPE_CPU';env=dict(os.environ,FP16_BMM_RUN_ROOT=str(root),FP16_BMM_GUARD_OWNER='fp16-bmm-registration1-recipe-cpu-only')
script=O/'capture_recipe.py';script.write_text("import json,os,pathlib\npathlib.Path(__file__).with_name('RECIPE_ENV.json').write_text(json.dumps(dict(os.environ),indent=2))\n")
result=subprocess.run(['bash',str(O/'run_environment.sh'),'/usr/bin/python3','-B','-S',str(script)],env=env,capture_output=True,text=True);assert result.returncode==0,result.stderr
observed=json.loads((O/'RECIPE_ENV.json').read_text());saved=dict(os.environ)
pth=pathlib.Path('/root/dyq/workspace/.venv/lib/python3.12/site-packages/fla_npu_opp_env.py')
assert {n.names[0].name for n in ast.walk(ast.parse(pth.read_text())) if isinstance(n,ast.Import)}=={'os'}
try:
 os.environ.clear();os.environ.update(observed)
 exec(compile(pth.read_bytes(),str(pth),'exec'),{'__file__':str(pth)})
 for k,v in constants['ENV'].items():assert os.environ[k]==v,(k,os.environ.get(k),v)
 final=dict(os.environ)
finally:os.environ.clear();os.environ.update(saved)
M=types.ModuleType('recipe_module');exec(compile(raw,'registration','exec'),M.__dict__)
assert final['LD_LIBRARY_PATH']==M.LD_PATH and final['PYTHONPATH']==M.PYTHON_PATH
assert all(pathlib.Path(final[k]).is_relative_to(root) for k in M.CACHE_NAMES)
# Existing or historic run roots fail before any SDK/cache writes.
assert subprocess.run(['bash',str(O/'run_environment.sh'),'false'],env=env,capture_output=True).returncode!=0
for name in ('fp16_local_remaining3','bf16_local_gemm18'):
 e=dict(env,FP16_BMM_RUN_ROOT=str(O.parent/name))
 assert subprocess.run(['bash',str(O/'run_environment.sh'),'false'],env=e,capture_output=True).returncode!=0
(O/'RECIPE_CHECKS.json').write_text(json.dumps({'passed':True,'checks':['bash syntax','original SDK shell exact LD/PYTHON paths','original .pth passive os/pathlib environment simulation matches runtime validator','all cache roots dedicated','existing and historical run roots rejected'],'venv_started':False,'device_imports':0,'npu_runs':0},indent=2)+'\n')
print('recipe checks passed')
