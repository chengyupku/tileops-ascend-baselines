from pathlib import Path
import subprocess,sys,json,time
T=Path(__file__).resolve().parent
rows=json.loads((T/'CASES.json').read_text());families={};results=[]
order=['square-256-bfloat16','qwen3-30b-ep8-empty-bfloat16','whisper-conv1-bias-bfloat16']
order += [r['key'][2] for r in sorted(rows,key=lambda r:r['envelope']['cost']) if r['key'][1]=='BmmFwdOp' and r['key'][2] not in order]
order+=['moe-fwd-bfloat16'];(T/'EXECUTION_ORDER.json').write_text(json.dumps(order,indent=2))
stop=None
for cid in order:
 r=next(x for x in rows if x['key'][2]==cid);op=r['key'][1];P=T/cid
 if stop or op in families:
  results.append({'key':r['key'],'status':'blocked','blocked_reason':stop or families[op],'executed':False});continue
 if not (P/'GUARD_RECEIPT.json').exists():
  subprocess.run([sys.executable,str(T/'run_stage.py'),cid,'300',str(P/'run.sh')],check=False)
 g=json.loads((P/'GUARD_RECEIPT.json').read_text());code=g['returncode']
 if code in [75,76,78]:stop=f'guard_{code}_stop_inspection_required at {cid}'
 if code==0:
  with (P/'validation.stdout.log').open('w') as out,(P/'validation.stderr.log').open('w') as err:
   validation=subprocess.run([sys.executable,str(T/'validate_case.py'),str(P)],stdout=out,stderr=err)
  if validation.returncode==0:
   results.append({'key':r['key'],**json.loads((P/'VALIDATION.json').read_text()),'evidence':str(P)});(T/'RESULT.partial.json').write_text(json.dumps(results,indent=2));continue
 reason=f'{cid}: guard_exit={code}; '
 if (P/'representative.json').exists():
  receipt=json.loads((P/'representative.json').read_text());reason+=json.dumps({k:receipt.get(k) for k in ['status','error','verification']})
 elif code==124:
  phase=json.loads((P/'PHASE.json').read_text()) if (P/'PHASE.json').exists() else {};reason+='timeout_300s phase='+str(phase.get('phase','before_observed_call'))+'; not a numerical failure'
 else:reason+='see guard.stderr.log / validation.stderr.log'
 if 'frozen source changed' in (P/'guard.stderr.log').read_text():stop='frozen pins changed; stop all execution'
 families[op]=reason
 results.append({'key':r['key'],'status':'blocked','blocked_reason':reason,'executed':True,'evidence':str(P),'guard_exit':code})
 (T/'RESULT.partial.json').write_text(json.dumps(results,indent=2))
(T/'RESULT.json').write_text(json.dumps({'accepted_added':0,'rows':results},indent=2))
print('COMPLETE',len(results),'verified',sum(r['status']=='verified_unregistered' for r in results),flush=True)
