from pathlib import Path
import json,hashlib,time,sys
T=Path(__file__).resolve().parent;P=T/'result'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
def dump(n,x):(T/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
rc=int(sys.argv[1]);receipt=read(P/'representative.json');execution=read(P/'EXECUTION.json');events=read(P/'NATIVE_CALLS.json') or []
lines=[]
for p in (T/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*'):
 if p.is_file():
  for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
   if any(x in line for x in ['GroupedMatmulV5','GroupedMatmul_e84aaec0eccf7b8304da51ea5f5a2076','StreamLaunch','Register']):lines.append(dict(file=str(p),line=i,text=line))
dump('NATIVE_LOG_EVIDENCE.json',lines)
pins=read(T/'FROZEN.json')['sha256']; after={p:sha(Path(p)) if Path(p).is_file() else None for p in pins};dump('PINS_AFTER.json',after)
changed={p:{'before':h,'after':after[p]} for p,h in pins.items() if h!=after[p]}
verified=bool(rc==0 and receipt and receipt.get('status')=='verified' and receipt.get('inputs_preserved') and not changed)
result=dict(status='verified_pending_parent_review' if verified else 'blocked_or_failed',accepted_added=0,registered=False,key=['ops-transformer','MoEGroupedGemmFwdOp','ds-v3-ep8-masked-bfloat16'],guard_exit_code=rc,checker=receipt,phase=read(P/'PHASE.json'),execution=execution,resource_admission=read(P/'RESOURCE_ADMISSION.json'),masked_observed=read(P/'MASKED_OBSERVED.json'),actual_native_function_entries=sum(e['event']=='enter' for e in events),pins_checked=len(pins),pins_changed=changed,logs_evidence_lines=len(lines),numerical_replays=0)
dump('RESULT.json',result)
dump('DELIVERY.json',dict(status=result['status'],accepted_added=0,registered=False,scope=result['key'],findings=result,checks=['CPU strict shape/slice contract','AST and launch positional args','frozen original source deployment SDK runtime','canonical full checker under dedicated guard'],remaining_issues=[] if verified else ['See guard logs, canonical receipt and last phase; no unchanged retry'],changed_files=[str(p.relative_to(T)) for p in T.iterdir() if p.is_file()]))
dump('HEARTBEAT.json',dict(stage='terminal',time=time.time(),status=result['status']))
dump('FILEHASHES.json',{str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'})
print(json.dumps({k:result[k] for k in ['status','guard_exit_code','actual_native_function_entries','pins_checked','pins_changed','logs_evidence_lines']}))
