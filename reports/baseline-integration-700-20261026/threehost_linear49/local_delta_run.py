from pathlib import Path
import os,sys,json,subprocess,time,hashlib
P=Path(__file__).resolve().parent;cid=sys.argv[1];label=sys.argv[2];D=P/label;D.mkdir(exist_ok=False)
assert cid in json.loads((P/'CHUNK_CASES.json').read_text())
r=json.loads((P/'package_local/RUNTIME_RECIPE.json').read_text());env=os.environ.copy();env.update(r['required_environment']);env.update(GDN_CASE_OWNER='baseline700-20261026/threehost_linear49',GDN_RUN_DIR=str(D),FLA_NPU_DISABLE_PTH='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',ASCEND_GLOBAL_LOG_LEVEL='0');env.pop('ASCEND_NPU_ARCH',None)
for n in ['TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
 (D/n).mkdir();env[n]=str(D/n)
env['PATH']='/usr/local/Ascend/cann-9.2.0-beta.2/tools/bisheng_compiler/bin:/usr/local/Ascend/cann-9.2.0-beta.2/compiler/ccec_compiler/bin:'+env.get('PATH','')
env['LD_LIBRARY_PATH']=':'.join(['/usr/local/Ascend/cann-9.2.0-beta.2/lib64',str(Path(r['opapi']).parent),'/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch/lib','/root/dyq/workspace/.venv/lib/python3.12/site-packages/torch_npu/lib','/usr/local/Ascend/driver/lib64/driver','/usr/local/Ascend/driver/lib64/common'])
cmd=['npu-guard','run','--owner',env['GDN_CASE_OWNER'],'--wait','120','--timeout','180' if cid=='prefill-3000-bfloat16' else ('300' if cid.startswith('decode') else '1200'),'--','/root/dyq/workspace/.venv/bin/python','-B',str(P/'package_local/entry.py'),'--op','DeltaNetFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(D/'checker.json')]
(D/'ENV.json').write_text(json.dumps({k:v for k,v in env.items() if k in ['PATH','LD_LIBRARY_PATH','GDN_CASE_OWNER','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_CUSTOM_OPP_PATH','ASCEND_OPP_PATH']},indent=2));(D/'FROZEN.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'package_local').iterdir() if f.is_file()},indent=2));t=time.time()
with (D/'stdout.log').open('w') as o,(D/'stderr.log').open('w') as e:rc=subprocess.run(cmd,env=env,cwd=D,stdout=o,stderr=e).returncode
(D/'GUARD_RESULT.json').write_text(json.dumps({'argv':cmd,'returncode':rc,'started_epoch':t,'finished_epoch':time.time(),'wall_seconds':time.time()-t,'metadata':json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())},indent=2));print(label,rc,flush=True)
