import subprocess,pathlib,json,time,sys,shutil
p=pathlib.Path(__file__).resolve().parent;label=sys.argv[1];out=p/label;out.mkdir(exist_ok=False)
for name in ['run_case.py','run_batch.py','native_log_capture.py','integration700_sgl_gdn_remote38.py','baseline-run']:
 shutil.copyfile(p/name,out/name)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_gdn_remote38','--wait','3600','--timeout','300','--','bash',str(p/'baseline-run'),'sgl-kernel-npu','python',str(p/'run_batch.py'),label]
s=time.time(); (out/'submitted.json').write_text(json.dumps({'epoch':s,'command':cmd}))
with (out/'guard.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
e=time.time();acqs=[json.loads(f.read_text())['utc_epoch'] for f in (p/'runs').glob(label+'--*/acquired.json')];acq=min(acqs) if acqs else None
receipt=dict(command=cmd,exit_code=r.returncode,start_epoch=s,end_epoch=e,total_s=e-s,queue_s=acq-s if acq else None,device_s=e-acq if acq else None)
(out/'GUARD_RESULT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));print((out/'guard.log').read_text()[-4000:])
