import pathlib,platform,sys,sysconfig,shutil,subprocess,json,os
W=pathlib.Path('/root/dyq/workspace') if platform.machine()=='x86_64' else pathlib.Path('/opt/tileops-all-pairs/workspace')
bases=[W/'.cache/ascend-classic/native-pilot-20261020/ops-nn/install',W/'reports/baseline-integration-700-20261026/threehost_nn11/vendor',W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build']
d={'architecture':platform.machine(),'python':sys.executable,'abi':sysconfig.get_config_var('SOABI'),'cwd':os.getcwd(),'compiler':{x:shutil.which(x) for x in ['c++','cmake','bisheng','ccec']},'bases':{},'sdk':{},'guard':None}
for b in bases:
 r={'exists':b.exists(),'libraries':[],'quant_files':[]}
 if b.exists():
  for f in b.rglob('*'):
   if 'install3' in f.parts or 'source' in f.parts or 'build' in f.parts:continue
   if f.is_file() and f.name=='libcust_opapi.so':
    q=subprocess.run(['nm','-D',str(f)],capture_output=True,text=True)
    r['libraries'].append({'path':str(f),'file':subprocess.run(['readelf','-h',str(f)],capture_output=True,text=True).stdout,'exports':[x for x in q.stdout.splitlines() if 'QuantMatmulV4' in x or 'QuantBatchMatmulV3' in x]})
   if 'quant' in f.name.lower() and ('matmul' in f.name.lower() or 'mat_mul' in f.name.lower()):r['quant_files'].append(str(f))
 r['quant_files']=r['quant_files'][:60];d['bases'][str(b)]=r
for root in ['/usr/local/Ascend/ascend-toolkit/latest','/usr/local/Ascend/cann']:
 q=pathlib.Path(root);d['sdk'][root]={'exists':q.exists(),'resolved':str(q.resolve())}
cmd=['npu-guard','status','--json'] if shutil.which('npu-guard') else ['python3',str(W/'tools/npu_guard.py'),'status','--json']
r=subprocess.run(cmd,capture_output=True,text=True);d['guard']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(d))
