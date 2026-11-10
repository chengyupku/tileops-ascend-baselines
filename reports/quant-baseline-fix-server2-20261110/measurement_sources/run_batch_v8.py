import pathlib,json,subprocess,sys,datetime
P=pathlib.Path(__file__).resolve().parent;H='openlibing-DevEnv-055409';R='/opt/quant-baseline-fix-server2-20261110'
manifest=pathlib.Path(sys.argv[1]).resolve();name=manifest.stem;jobs=json.loads(manifest.read_text());out=P/'batches'/name;out.mkdir(parents=True,exist_ok=True)
with subprocess.Popen(['ssh','-o','ClearAllForwardings=yes',H,f'cat > {R}/{manifest.name}'],stdin=subprocess.PIPE) as s:
 s.communicate(manifest.read_bytes());assert s.returncode==0
cmd=f'env LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64 /opt/tileops-ci/runtime/bin/python -I -S /opt/torch-v010-20261106/tools/npu_guard.py run --owner quant-baseline/{name} --wait 3600 --timeout 900 --receipt {R}/{name}-guard.json -- /usr/bin/python3 {R}/device_batch_v8.py {R}/{manifest.name}'
print('BATCH_START',name,flush=True)
with (out/'run.log').open('w') as f:r=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=20',H,cmd],stdout=f,stderr=subprocess.STDOUT)
(out/'exit.json').write_text(json.dumps({'exit_code':r.returncode,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
files=[manifest.name,name+'.progress.json',name+'-guard.json']
for j in jobs:files.extend([f"runs/{j['index']}/{j['variant']}/{j['attempt']}",f"{j.get('mode','pilot')}-{j['index']}-{j['variant']}-{j['attempt']}.log"])
with (out/'fetch.log').open('w') as f:
 s=subprocess.Popen(['ssh','-o','ClearAllForwardings=yes',H,f"cd {R} && tar -cf - "+' '.join(files)],stdout=subprocess.PIPE,stderr=f)
 d=P/'remote';d.mkdir(exist_ok=True);a=subprocess.run(['tar','-C',str(d),'-xf','-'],stdin=s.stdout,stderr=f);s.stdout.close();rc=s.wait()
(out/'fetch.json').write_text(json.dumps({'ssh':rc,'tar':a.returncode}))
print('BATCH_DONE',name,r.returncode,flush=True)
