import pathlib,subprocess,time,json,sys,datetime
D=pathlib.Path(__file__).resolve().parent;case=sys.argv[1];C=D/case
assert not C.exists(), 'No unchanged retry'
for x in ['tmp','xdg','work','cache','logs','extensions','triton']:(C/x).mkdir(parents=True,exist_ok=True)
cmd=['npu-guard','run','--owner','baseline700-20261026/causal_native_build2','--wait','600','--timeout','300','--','bash',str(D/'check_child.sh'),case]
t=time.time()
with (C/'stdout.log').open('w') as out,(C/'stderr.log').open('w') as err:p=subprocess.run(cmd,stdout=out,stderr=err)
r=dict(argv=cmd,start_epoch=t,end_epoch=time.time(),elapsed_seconds=time.time()-t,returncode=p.returncode)
if (C/'guard_running.json').exists():
 m=json.loads((C/'guard_running.json').read_text());r['running_metadata']=m
 start=datetime.datetime.fromisoformat(m['requested_at']).timestamp();acq=datetime.datetime.fromisoformat(m['acquired_at']).timestamp();run=datetime.datetime.fromisoformat(m['updated_at']).timestamp();r.update(pure_lockwait_seconds=acq-start,idle_admission_seconds=run-acq,execution_seconds=time.time()-run)
(C/'GUARD_RECEIPT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if (C/'checker.json').exists():
 j=json.loads((C/'checker.json').read_text());print(json.dumps({k:j.get(k) for k in ['status','reason','stage','verification']}))
