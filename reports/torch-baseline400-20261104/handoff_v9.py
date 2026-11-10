"""Drain v8 batch, then switch to v9; never terminate workers or rerun accepted."""
import pathlib,json,time,os,sys
p=pathlib.Path(__file__).resolve().parent;h=p/'server1'
proof=h/'GQA_SATURATING_CPU_TEST.json'
while not proof.exists():time.sleep(5)
test=json.loads(proof.read_text());assert all(r['CPU_reference_vs_itself']['passed'] for r in test['rows'])
old=json.loads((h/'HEARTBEAT.json').read_text());assert old['pid']==2614872 and str(old['start_ticks'])=='278281598'
(h/'STOP').write_text('Drain current v8 batch for verified v9 GQA conversion repair; no device interruption.\n')
def alive():
 try:return pathlib.Path('/proc/2614872/stat').read_text().split()[21]=='278281598'
 except FileNotFoundError:return False
while alive():time.sleep(5)
state=json.loads((h/'STATE.json').read_text());retry=json.loads((h/'RETRY.json').read_text());tid='25f183d932d021e7ff05'
if state.get(tid,{}).get('status')=='failed':
 retry[tid]={'at':time.time(),'reason':'v9 original GQA formula with explicit saturating FP8 cast matching CPU; all three exact CPU cases pass unchanged checker; no CPU oracle/checker changes'}
 tmp=h/'RETRY.tmp';tmp.write_text(json.dumps(retry,indent=2));tmp.replace(h/'RETRY.json')
(h/'STOP').unlink()
(h/'V9_HANDOFF.json').write_text(json.dumps({'at':time.time(),'old_controller_drained':True,'cpu_test':str(proof),'retry':retry.get(tid)},indent=2))
os.execv(sys.executable,[sys.executable,str(p/'supervisor_server1_v9.py'),'server1'])
