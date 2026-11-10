import subprocess,pathlib,json,time,sys
p=pathlib.Path(__file__).resolve().parent
op,cid,label=sys.argv[1:];out=p/'runs'/(label+'--'+op+'--'+cid);out.mkdir(parents=True,exist_ok=False)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_linalg36','--wait','3600','--timeout','300','--','bash',str(p/'baseline-run'),'sgl-kernel-npu','python',str(p/'run_case.py'),op,cid,label]
s=time.time(); (out/'submitted.json').write_text(json.dumps({'epoch':s,'command':cmd}))
with (out/'guard.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
e=time.time();acq=json.loads((out/'acquired.json').read_text())['utc_epoch'] if (out/'acquired.json').exists() else None
receipt=dict(command=cmd,exit_code=r.returncode,start_epoch=s,end_epoch=e,total_s=e-s,queue_s=acq-s if acq else None,device_s=e-acq if acq else None)
(out/'GUARD_RESULT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));print((out/'guard.log').read_text()[-3000:])
