import subprocess,sys,pathlib,time,json
base=pathlib.Path(__file__).parent
name=sys.argv[1]; script=sys.stdin.read(); (base/f'{name}.sh').write_text(script)
t=time.time()
with (base/f'{name}.log').open('w') as f:
 p=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','bash -s'],input=script,text=True,stdout=f,stderr=subprocess.STDOUT)
(base/f'{name}.receipt.json').write_text(json.dumps(dict(exit_code=p.returncode,elapsed=time.time()-t),indent=2))
print((base/f'{name}.log').read_text()[-25000:]); print('RECEIPT',p.returncode,time.time()-t)
