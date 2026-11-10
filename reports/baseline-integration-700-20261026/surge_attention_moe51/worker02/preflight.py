import pathlib, subprocess, json, platform, sys, hashlib, datetime
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arch':platform.machine(),'python':sys.version,'paths':{}}
for name in ['/etc/hccl_rootinfo.json','/usr/local/Ascend/driver/topo/950','/opt/tileops-all-pairs/bin/allpairs-python','/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/deep_ep/deep_ep/strategies/normal_strategy.py']:
 p=pathlib.Path(name); d={'exists':p.exists(),'is_file':p.is_file(),'is_dir':p.is_dir()}
 if p.is_dir(): d['entries']=[x.name for x in p.iterdir()]
 if p.is_file(): d['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 out['paths'][name]=d
out['rootinfo_mount']=[x for x in pathlib.Path('/proc/mounts').read_text().splitlines() if 'hccl_rootinfo' in x]
r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True,timeout=30);out['guard_status']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out,indent=2))
