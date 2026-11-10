import pathlib,json,subprocess,datetime,concurrent.futures
P=pathlib.Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=P/'inspections'/stamp;out.mkdir(parents=True)
hosts={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}
def inspect(h):
 H=P/h;state=json.load(open(H/'STATE.json'));pid=state['pid'];proc=pathlib.Path('/proc')/str(pid)
 ticks=proc.joinpath('stat').read_text().split()[21] if proc.exists() else None
 cmd="env LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64 /opt/tileops-ci/runtime/bin/python -I -S /opt/torch-v010-20261106/tools/npu_guard.py status --json"
 r=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15',hosts[h],cmd],capture_output=True,text=True,timeout=60);guard=json.loads(r.stdout)
 disk_cmd="python3 -c 'import os,json; s=os.statvfs(\"/opt\"); print(json.dumps({\"available_bytes\":s.f_bavail*s.f_frsize,\"total_bytes\":s.f_blocks*s.f_frsize}))'"
 disk_result=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15',hosts[h],disk_cmd],capture_output=True,text=True,timeout=30)
 disk_result.check_returncode();disk=json.loads(disk_result.stdout);disk.update(floor_bytes=50*1024**3,target_bytes=60*1024**3,below_floor=disk['available_bytes']<50*1024**3,maintenance_needed=disk['available_bytes']<60*1024**3)
 pr=json.load(open(H/'PROGRESS.json')) if (H/'PROGRESS.json').exists() else {};plan=json.load(open(H/'PLAN.json'))['rows'];ids={x['id'] for x in plan};accepted={k for k,v in pr.items() if k in ids and v.get('status')=='accepted'};pending={k:v for k,v in pr.items() if k in ids and v.get('status')!='accepted'}
 done=[f for f in (H/'runs').glob('*/a*/DEVICE_DONE.json') if not (f.parent/'ARCHIVE.json').exists()]
 result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_identity_matches':ticks==str(state.get('start_ticks')),'state':state,'guard':guard,'disk':disk,'accepted':len(accepted),'total':len(plan),'pending':pending,'unarchived_attempts':[str(f.parent) for f in done],'journal_tail':(H/'journal.jsonl').read_text().splitlines()[-8:]};(out/(h+'.json')).write_text(json.dumps(result,indent=2));return h,result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:d=dict(pool.map(inspect,hosts))
(P/'LATEST_INSPECTION.json').write_text(json.dumps({'path':str(out),'counts':{h:{k:v[k] for k in ['accepted','total','controller_identity_matches','unarchived_attempts','disk']} for h,v in d.items()}},indent=2));print((P/'LATEST_INSPECTION.json').read_text())
