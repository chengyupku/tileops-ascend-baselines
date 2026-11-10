import subprocess,sys,pathlib,time,json
P=pathlib.Path(__file__).parent
host,name=sys.argv[1:3];code=sys.stdin.read();(P/(name+'.input')).write_text(code)
t=time.time();r=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh',host,'python3 -'],input=code,text=True,capture_output=True)
(P/(name+'.log')).write_text(r.stdout+r.stderr);(P/(name+'.receipt.json')).write_text(json.dumps(dict(rc=r.returncode,elapsed=time.time()-t)))
print(r.stdout[-16000:]);print(r.stderr[-2000:]);sys.exit(r.returncode)
