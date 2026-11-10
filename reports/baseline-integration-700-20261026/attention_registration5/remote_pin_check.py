import pathlib,json,subprocess
P=pathlib.Path(__file__).resolve().parent
pins=json.loads((P/'PINS.json').read_text())
code='import pathlib,hashlib,json\npins='+repr(pins)+'''\nresult={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in pins.items()}\nprint(json.dumps(result))\n'''
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([ssh,'openlibing-DevEnv-055409','python3 -'],input=code,text=True,capture_output=True,timeout=45)
(P/'SERVER2_CPU_PIN_CHECK.json').write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'cpu_only':True},indent=2))
print(r.returncode,r.stdout[-300:],r.stderr)
