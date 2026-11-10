import os,json,subprocess,sys,time,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;cid=sys.argv[1];label=sys.argv[2];D=P/label;D.mkdir(exist_ok=False)
assert cid in json.loads((P/'CHUNK_CASES.json').read_text())
env=os.environ.copy();env.update(GDN_RUN_DIR=str(D),GDN_CASE_OWNER='baseline700-20261026/threehost_linear49',ASCEND_HOME_PATH='/usr/local/Ascend/cann-9.1.0',ASCEND_TOOLKIT_HOME='/usr/local/Ascend/cann-9.1.0',ASCEND_OPP_PATH='/usr/local/Ascend/cann-9.1.0/opp',TORCH_DEVICE_BACKEND_AUTOLOAD='0',FLA_NPU_DISABLE_PTH='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',ASCEND_GLOBAL_LOG_LEVEL='0')
env.pop('ASCEND_CUSTOM_OPP_PATH',None)
env['PATH']='/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin:/usr/local/Ascend/cann-9.1.0/compiler/ccec_compiler/bin:'+env.get('PATH','')
for n in ['TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 (D/n).mkdir();env[n]=str(D/n)
W=P.parents[2];env['LD_LIBRARY_PATH']=':'.join([str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch/lib'),str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch_npu/lib'),'/usr/local/Ascend/cann-9.1.0/lib64','/usr/local/Ascend/driver/lib64/common','/usr/local/Ascend/driver/lib64/driver'])
if len(sys.argv)>3 and sys.argv[3]=='runtime92':
 r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text())
 env['LD_LIBRARY_PATH']=':'.join(r['ld_library_path_order'])+':'+str(Path(r['opapi']).parent)
 env['ASCEND_OPP_PATH']=r['required_environment']['ASCEND_OPP_PATH']
 env['ASCEND_CUSTOM_OPP_PATH']=r['vendor']
 (P/'package/RUNTIME_RECIPE.json').write_text(json.dumps({'opapi':r['opapi'],'nnopbase':r['nnopbase']}))
if len(sys.argv)>3 and sys.argv[3]=='custom92_runtime91':
 r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text())
 env['ASCEND_CUSTOM_OPP_PATH']=r['vendor']
 env['LD_LIBRARY_PATH']=str(Path(r['opapi']).parent)+':'+env['LD_LIBRARY_PATH']
 (P/'package/RUNTIME_RECIPE.json').write_text(json.dumps({'opapi':r['opapi'],'nnopbase':'/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so'}))
package_name='package'
if len(sys.argv)>3 and sys.argv[3]=='recurrent91':
 package_name='package_recurrent'
 r=json.loads((P/package_name/'RUNTIME_RECIPE.json').read_text())
 env['ASCEND_CUSTOM_OPP_PATH']=r['vendor']
 env['LD_LIBRARY_PATH']=str(Path(r['opapi']).parent)+':'+env['LD_LIBRARY_PATH']
if len(sys.argv)>3 and sys.argv[3]=='packed91':
 package_name='package_packed'
 r=json.loads((P/'ORIGINAL92_RECIPE.json').read_text())
 env['ASCEND_CUSTOM_OPP_PATH']=r['vendor']
 env['LD_LIBRARY_PATH']=str(Path(r['opapi']).parent)+':'+env['LD_LIBRARY_PATH']
 (P/package_name/'RUNTIME_RECIPE.json').write_text(json.dumps({'opapi':r['opapi'],'nnopbase':'/usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so'}))
cmd=['npu-guard','run','--owner',env['GDN_CASE_OWNER'],'--wait','120','--timeout',('300' if cid.startswith('decode') else '1200'),'--','/opt/tileops-ci/runtime/bin/python','-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py',str(P/package_name/'entry.py'),'--op','DeltaNetFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(D/'checker.json')]
(D/'ENV.json').write_text(json.dumps({k:v for k,v in env.items() if k in ['LD_LIBRARY_PATH','GDN_CASE_OWNER','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_OPP_PATH']},indent=2));start=time.time()
with (D/'stdout.log').open('w') as out,(D/'stderr.log').open('w') as err:
 r=subprocess.run(cmd,env=env,cwd=D,stdout=out,stderr=err)
(D/'GUARD_RESULT.json').write_text(json.dumps({'argv':cmd,'returncode':r.returncode,'started_epoch':start,'finished_epoch':time.time(),'wall_seconds':time.time()-start,'queue_seconds':None,'physical_wait_seconds':None,'metadata':json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())},indent=2));print(label,r.returncode,flush=True)
