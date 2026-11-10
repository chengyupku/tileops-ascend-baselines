import ctypes as C,json,socket,time,os
from pathlib import Path
h=C.CDLL('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so');f=h.halGetDeviceInfo;f.argtypes=[C.c_uint,C.c_int,C.c_int,C.POINTER(C.c_int64)];f.restype=C.c_int
hw={}
for module,name in [(4,'aicore'),(7,'vectorcore')]:
 v=C.c_int64();rc=f(0,module,3,C.byref(v));hw[name]={'rc':rc,'count':v.value}
roots=[Path('/root/dyq/workspace/reports'),Path('/opt/tileops-baseline-cache/workspace/reports'),Path('/opt/tileops-all-pairs/workspace/reports')]
files=[];rows=[];errors=[]
for root in roots:
 if not root.exists():continue
 for folder,dirs,names in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ('archives','node_modules','.git','__pycache__')]
  if 'CACHE.jsonl' not in names:continue
  p=Path(folder)/'CACHE.jsonl';files.append(str(p))
  for line_no,line in enumerate(p.open(),1):
   try:r=json.loads(line)
   except Exception as ex:errors.append([str(p),line_no,str(ex)]);continue
   env=r.get('identity',{}).get('environment',{})
   rows.append({k:r.get(k) for k in ['key','task_id','status','cache_key','performance_valid','started_at','finished_at','correctness_receipt','guard_receipt','cohort_key']}|{'hostname':env.get('hostname'),'device_name':env.get('device_name'),'platform':env.get('platform'),'source_file':str(p),'source_line':line_no,'latency_ms':r.get('timing',{}).get('latency_ms')})
print(json.dumps({'at':time.time(),'hostname':socket.gethostname(),'hardware':hw,'files':files,'rows':rows,'errors':errors}))
