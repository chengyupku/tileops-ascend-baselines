"""Hold the existing provider coordination lock read-only during one guarded check."""
from pathlib import Path
import os,sys,fcntl,json,datetime,hashlib
P=Path(__file__).resolve().parent;label=sys.argv[1];lock=Path('/tmp/baseline700-provider-sgl.lock');fd=os.open(lock,os.O_RDONLY)
fcntl.flock(fd,fcntl.LOCK_SH);os.set_inheritable(fd,True)
out=P/'provider_locks';out.mkdir(exist_ok=True)
provider=P.parents[2]/'TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'
(out/(label+'.json')).write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lock':str(lock),'mode':'read-only descriptor, shared flock','fd':fd,'inherited_across_exec':True,'created_lock':False,'provider_sha256':hashlib.sha256(provider.read_bytes()).hexdigest(),'command':sys.argv[2:]},indent=2))
os.execvp(sys.argv[2],sys.argv[2:])
