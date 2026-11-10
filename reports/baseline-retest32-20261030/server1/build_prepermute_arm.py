"""CPU-only original host/source package build with private disk guard."""
from pathlib import Path
import os,subprocess,time,json,shutil,signal
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1');b=h/'native_build/selection_v1';s=b/'source';assert s.is_dir()
assert shutil.disk_usage(b).free>50.5*2**30
cmd=['bash','-c','source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexport PYTHONPATH=/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1/native_build/selection_v1/python_deps\nexport PYTHONDONTWRITEBYTECODE=1\nexec bash build.sh --pkg --jit --soc=ascend950 --ops=moe_init_routing_v2 --vendor_name=prepermutepairs -j8 --build-type=Release --ccache off']
row={'started_at':time.time(),'command':cmd,'cwd':str(s),'cpu_only':True,'original_build_flags':True,'minimum_free_bytes':50*2**30}
with (b/'prepermute_build.log').open('w') as f:
 p=subprocess.Popen(cmd,cwd=s,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 row['pid']=p.pid
 while p.poll() is None:
  free=shutil.disk_usage(b).free
  if free<50.5*2**30 or time.time()-row['started_at']>1200:
   row['stopped_reason']='disk_headroom' if free<50.5*2**30 else 'timeout';os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);break
  time.sleep(2)
 row.update(returncode=p.returncode,finished_at=time.time(),free_bytes=shutil.disk_usage(b).free)
(b/'prepermute_build_receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
