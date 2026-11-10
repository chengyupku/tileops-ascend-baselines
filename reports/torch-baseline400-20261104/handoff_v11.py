import pathlib,json,time,os,sys
p=pathlib.Path(__file__).resolve().parent;h=p/'server1'
while True:
 try:alive=pathlib.Path('/proc/3747371/stat').read_text().split()[21]=='281152853'
 except FileNotFoundError:alive=False
 if not alive:break
 time.sleep(1)
s=json.loads((h/'STATE.json').read_text());retry=json.loads((h/'RETRY.json').read_text());ids=[]
for tid,v in s.items():
 d=h/'runs'/tid/v['attempt'];f=d/'prepare_failure.json'
 memory_failure=f.exists() and 'CPU memory admission unavailable' in f.read_text()
 if v['status']=='cancelled' or memory_failure:
  retry[tid]={'at':time.time(),'reason':'v11 CPU-only MoE admission: serialize entire prepare with host file lock, estimate2.2x inputs+8GiB instead impossible2.8x; original generator/reference/checker/deviceformula unchanged. Cancelled waiting prepares may resume under repaired resource admission.'};ids.append(tid)
t=h/'RETRY.tmp';t.write_text(json.dumps(retry,indent=2));t.replace(h/'RETRY.json')
(h/'V11_HANDOFF.json').write_text(json.dumps({'at':time.time(),'retry_ids':ids,'old_pid_exited':True,'change':'CPU memory admission only'},indent=2));(h/'STOP').unlink(missing_ok=True)
os.execv(sys.executable,[sys.executable,str(p/'supervisor_server1_v11.py'),'server1'])
