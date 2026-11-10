from pathlib import Path
import json,hashlib,time
T=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
def dump(n,x):(T/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
all_changes={}; pin_count=0
rows=[]
for case in read(T/'CASES.json'):
 C=T/case['key'][2];P=C/'result';
 pins=read(C/'FROZEN.json')['sha256'];after=read(T/'PINS_AFTER_PREPUBLICATION.json') if case['key'][2]=='moe-dgrad-float16' else {p:sha(Path(p)) if Path(p).is_file() else None for p in pins}
 (C/'PINS_AFTER.json').write_text(json.dumps(after,indent=2));changed={p:dict(before=h,after=after.get(p)) for p,h in pins.items() if after.get(p)!=h};all_changes[case['key'][2]]=changed;pin_count+=len(pins)
 rc=int((C/'guard.exit').read_text()) if (C/'guard.exit').exists() else None;receipt=read(P/'representative.json');events=read(P/'NATIVE_CALLS.json') or [];phase=read(P/'PHASE.json');lines=[]
 for p in (C/'runtime/ASCEND_PROCESS_LOG_PATH').rglob('*'):
  if p.is_file():
   for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
    if any(x in line for x in ['GroupedMatmulV5','GroupedMatmul_e84aaec0eccf7b8304da51ea5f5a2076','StreamLaunch','Register']):lines.append(dict(file=str(p),line=i,text=line))
 (C/'NATIVE_LOG_EVIDENCE.json').write_text(json.dumps(lines,indent=2))
 entries=[e for e in events if e['event']=='enter'];native_ok=bool(len(entries)>=2 and all(e['map_inode']==e['stat_inode'] and e['path'].endswith('/bf16gmm_transformer/op_api/lib/libcust_opapi.so') for e in entries) and any('e84aaec0eccf7b8304da51ea5f5a2076' in x['text'] and 'Register' in x['text'] for x in lines) and any('e84aaec0eccf7b8304da51ea5f5a2076' in x['text'] and 'StreamLaunch' in x['text'] for x in lines))
 verified=native_ok and rc==0 and receipt and receipt.get('status')=='verified' and receipt.get('inputs_preserved') and not changed
 status='verified_pending_parent_review' if verified else 'reference_timeout_candidate_not_run' if rc==124 and phase and phase['phase']=='reference' and not events else 'unrun' if rc is None else 'failed_or_blocked'
 rows.append(dict(key=case['key'],status=status,guard_exit_code=rc,checker=receipt,phase=phase,timings=read(P/'TIMINGS.json'),native_verified=native_ok,native_entries=sum(e['event']=='enter' for e in events),native_evidence_lines=len(lines),execution=read(P/'EXECUTION.json'),resource_admission=read(P/'RESOURCE_ADMISSION.json')))
dump('RESULT.json',dict(accepted_added=0,registered=False,rows=rows,pins_checked=pin_count,pins_changed=all_changes))
dump('DELIVERY.json',dict(accepted_added=0,registered=False,rows=rows,checks=['CPU launch AST/positional args and FP16 metadata contracts','Original canonical checker with full output/input preservation under guard','Actual V5 function addresses and maps plus runtime device logs','Before and after source/native/deployment/runtime pins'],remaining_issues=['Parent fullchecker/native review and formal exact registry publication required for verified keys.']+[r['key'][2]+': '+r['status'] for r in rows if r['status']!='verified_pending_parent_review'],changed_files=[str(p.relative_to(T)) for p in T.iterdir() if p.is_file()]))
dump('HEARTBEAT.json',dict(time=time.time(),stage='finalized',rows=[dict(key=r['key'],status=r['status']) for r in rows]))
dump('FILEHASHES.json',{str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name!='FILEHASHES.json'})
print(json.dumps([dict(key=r['key'],status=r['status'],rc=r['guard_exit_code'],native=r['native_entries']) for r in rows]))
