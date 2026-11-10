import pathlib,subprocess,sys,json
p=pathlib.Path(__file__).resolve().parent
for n in (90,91):
 subprocess.run([sys.executable,str(p/'run_batch_v15.py'),str(p/f'batch{n}.json')],check=True)
 g=json.loads((p/f'remote/batch{n}-guard.json').read_text())
 assert g.get('performance_valid') and g.get('exit_code')==0,g
 subprocess.run([sys.executable,str(p/'audit_results.py')],check=True)
 for j in json.loads((p/f'batch{n}.json').read_text()):
  m=json.loads((p/f"remote/runs/{j['index']}/ops_math/formal1/result.json").read_text())
  assert 'timing' in m,m
print('OPS_MATH_FORMAL_DONE',flush=True)
