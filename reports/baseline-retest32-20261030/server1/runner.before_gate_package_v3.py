"""CPU supervisor for isolated 32-core remeasurement, existing pipeline/guard."""
import json,os,time,fcntl,sys
from pathlib import Path
from baseline_retest32_server1_pipeline import run
from baseline_cache_guard_binding import capture_guard,bind_guard
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'reports/baseline-retest32-20261030'; H=C/'server1'; H.mkdir(parents=True,exist_ok=True)
def read(p,default=None):
    return json.loads(p.read_text()) if p.exists() else default
def put(p,x):
    q=p.with_suffix('.tmp'); q.write_text(json.dumps(x,ensure_ascii=False,indent=2)); q.replace(p)
def append(p,x):
    with p.open('a') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
lock=(H/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
identity={'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
while not (H/'STOP').exists():
    all_tasks=read(C/'server1_ASSIGNMENTS.json')['pairs']
    all_tasks.sort(key=lambda t: (t['key'][0]!='flash-linear-attention-npu',t['key'][1]!='AddFwdOp',t['expected_seconds']))
    hist=rows(H/'journal.jsonl')
    latest={x['id']:x for x in hist}
    def resolved_path_retry(x):
        return x.get('revision')=='native_v1' and x.get('error')=="No module named 'fla_npu'"
    attempted={x['id'] for x in latest.values() if x['status']=='cached' or not resolved_path_retry(x)}
    holds={tuple(x['key'][:2]) for x in latest.values() if x['status']!='cached' and not resolved_path_retry(x)}
    # Only proven remote deployments are released. Others require CPU inspection.
    enabled=read(H/'ENABLED.json',{'libraries':['flash-linear-attention-npu','sgl-kernel-npu','DeepGEMM-Ascend']})['libraries']
    todo=[t for t in all_tasks if t['id'] not in attempted and t['key'][0] in enabled and tuple(t['key'][:2]) not in holds]
    put(H/'STATE.json',{'at':time.time(),'identity':identity,'phase':'preparing' if todo else 'needs_review','assigned':len(all_tasks),'cached':sum(x['status']=='cached' for x in latest.values()),'pending_review':sum(x['status']!='cached' for x in latest.values()),'unattempted':len(all_tasks)-len(attempted),'enabled_libraries':enabled,'held_operator_scopes':[list(k) for k in holds]})
    if not todo: break
    batch=H/'batches'/str(time.time_ns()); jobs=[]
    chosen=[]; scopes=set()
    for t in todo:
        scope=tuple(t['key'][:2])
        if scope in scopes:continue
        scopes.add(scope);chosen.append(t)
        if len(chosen)==4:break
    for t in chosen:
        out=H/'pairs'/t['id']/str(time.time_ns());out.mkdir(parents=True)
        put(out/'task.json',t)
        jobs.append({'id':t['id'],'prepare':{'cpu_only':True,'timeout':120,'argv':['python3',str(ROOT/'tools/baseline_retest32_server1_prepare.py'),str(out)]},'device':{'timeout':t['timeout'],'argv':['bash',str(ROOT/'tools/baseline_retest32_server1_python.sh'),t['key'][0],str(ROOT/'tools/measure_baseline_retest32_server1.py'),'--task',str(out/'task.json'),'--out',str(out)]},'out':str(out),'key':t['key']})
    put(H/'ACTIVE.json',{'batch':str(batch),'jobs':jobs,'at':time.time(),'identity':identity})
    run({'owner':'baseline-retest32-20261030/server1','wait':600,'cwd':str(ROOT),'jobs':jobs},batch,lookahead=2,guard_command=['python3',str(ROOT/'tools/baseline_retest32_server1_guard_receipt.py')])
    for job in jobs:
        out=Path(job['out']); guardpath=batch/job['id']/'guard.json'; g=read(guardpath,{})
        if not g:
            if (batch/job['id']/'device.log').exists():
                put(H/'STOP',{'reason':'Device launcher failed without receipt; manual inspection required'})
            continue # Prepared-only jobs were never attempted on device.
        m=read(out/'measurement.json',{}); status='pending_review'; err=m.get('error')
        if m.get('status')=='measured_pending_guard' and g.get('performance_valid') is True and g.get('workload_exit_code')==0:
            m.update(status='cached',performance_valid=True,guard_receipt=str(guardpath))
            try:
                m['guard_binding']=bind_guard(m,read(out/'prepare.json')['guard_before'],capture_guard())
                assert m['identity']['environment']['aicore_count']==32
                put(out/'measurement.json',m);append(H/'CACHE.jsonl',m);status='cached'
            except Exception as e: err=str(e)
        append(H/'journal.jsonl',{'id':job['id'],'key':job['key'],'at':time.time(),'status':status,'measurement':str(out/'measurement.json'),'guard':str(guardpath),'error':err,'batch':str(batch),'revision':'package_path_v2'})
