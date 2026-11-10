import subprocess,json,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text());ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';D='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_linear49';v=Path(r['vendor']);t=time.time()
a=subprocess.run([ssh,'openlibing-DevEnv-055409',f'tar -czhf - -C {v.parent} {v.name}'],capture_output=True,check=True);(P/'native_custom92_aarch64.tar.gz').write_bytes(a.stdout)
subprocess.run([ssh,'openlibing-DevEnv-661495',f'mkdir -p {D}/native_custom92 && tar -xzf - -C {D}/native_custom92'],input=a.stdout,check=True)
r['vendor']=D+'/native_custom92/'+v.name;r['opapi']=r['vendor']+'/op_api/lib/libcust_opapi.so'
for n,data in [('ORIGINAL92_RECIPE.json',json.dumps(r,indent=2).encode()),('remote_run.py',(P/'remote_run.py').read_bytes())]:subprocess.run([ssh,'openlibing-DevEnv-661495',f'cat > {D}/{n}'],input=data,check=True)
(P/'MINIMAL_MIGRATION_RECEIPT.json').write_text(json.dumps({'from':'server2','to':'server1','original_vendor':str(v),'private_vendor':r['vendor'],'same_arch':'aarch64','archive_sha256':hashlib.sha256(a.stdout).hexdigest(),'seconds':time.time()-t,'purpose':'match original API descriptor allocator with actually linked SDK9.1 libnnopbase; unchanged native binaries'},indent=2));print('minimal ARM native migration complete',flush=True)
