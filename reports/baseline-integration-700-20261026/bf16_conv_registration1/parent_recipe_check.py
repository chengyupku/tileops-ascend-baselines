import types,pathlib,os,json
p=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/bf16_conv_registration1/integration700_bf16_conv_registration1.py');m=types.ModuleType('test');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
expected=dict(m.ENV,LD_LIBRARY_PATH=m.LD_PATH,PYTHONPATH=m.PYTHON_PATH)
assert all(os.environ.get(k)==v for k,v in expected.items()), {k:os.environ.get(k) for k,v in expected.items() if os.environ.get(k)!=v}
print(json.dumps(dict(passed=True,actual_shell_environment_matches=True,device_imports=0)))
