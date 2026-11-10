import pathlib,subprocess,sys,json,time,datetime,hashlib,platform
p=pathlib.Path(__file__).resolve().parent;cid=sys.argv[1];label='exact01';out=p/'runs'/(label+'--GDNFwdOp--'+cid);out.mkdir(parents=True,exist_ok=True)
root=pathlib.Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source/python/sgl_kernel_npu/sgl_kernel_npu/fla')
files=list(root.glob('*.py'))+[p/n for n in ['run_case.py','baseline-run','integration700_sgl_gdn_remote38.py','route_install.py','CHECKER_PIN.json']]+[pathlib.Path('/opt/tileops-all-pairs/workspace/TileOPs/scripts/ascend_pair_check.py'),pathlib.Path('/opt/tileops-all-pairs/workspace/TileOPs/workloads/numerics.py'),pathlib.Path('/opt/tileops-all-pairs/bin/allpairs-python')]
(out/'IDENTITY_BEFORE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arch':platform.machine(),'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2))
cmd=['npu-guard','run','--owner','baseline700-20261026/surge_dense_linear92/worker07','--wait','120','--timeout','600','--','bash',str(p/'baseline-run'),'sgl-kernel-npu','python',str(p/'run_case.py'),'GDNFwdOp',cid,label]
s=time.time();(out/'REQUEST.json').write_text(json.dumps({'epoch':s,'command':cmd}))
with (out/'guard.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
e=time.time();a=out/'acquired.json';ac=json.loads(a.read_text()) if a.exists() else None
receipt={'command':cmd,'requested_epoch':s,'end_epoch':e,'exit_code':r.returncode,'metadata':ac,'wall_seconds':e-s,'queue_and_idle_seconds':ac['utc_epoch']-s if ac else None,'child_wall_seconds':e-ac['utc_epoch'] if ac else None,'physical_wait_seconds':None,'held_cpu_seconds':None,'device_execution_seconds':None}
(out/'GUARD_RESULT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
