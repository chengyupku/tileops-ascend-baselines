"""Private deployment recipe. CPU-only launcher; device work is wholly inside npu-guard."""
import sys,os,json,pathlib,subprocess,datetime,platform
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import registration as M
cid=sys.argv[1];pin=M.metadata()['RUNTIME_PINS.json'][cid]
if platform.machine()!=pin['host']['architecture'] or platform.node()!=pin['host']['hostname']:raise SystemExit('Wrong verified host for this key')
root=P/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');root.mkdir(parents=True,exist_ok=False);(root/'RUN_EXCLUSIVE').touch()
env=os.environ.copy();env.update(pin['environment']);env.update(GDN_CASE_OWNER='baseline700-20261026/threehost_linear_registration3',LINEAR3_RUN_ROOT=str(root),PYTHONDONTWRITEBYTECODE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',FLA_NPU_DISABLE_PTH='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',ASCEND_GLOBAL_LOG_LEVEL='0')
for k in ('LD_PRELOAD','PYTHONHOME','ASCEND_NPU_ARCH'):env.pop(k,None)
for k in M.CACHE_NAMES:(root/k).mkdir();env[k]='/proc/self/cwd/'+k
if 'PATH' not in pin['environment']:
 sdk=pin['environment']['ASCEND_HOME_PATH'];env['PATH']=sdk+'/tools/bisheng_compiler/bin:'+sdk+'/compiler/ccec_compiler/bin:'+env.get('PATH','')
python=pin['python_prefix']+'/bin/python'
entry=[python,'-B',str(P/'runtime_entry.py'),cid,str(root/'checker.json')]
if pin['host']['architecture']=='aarch64':entry=[python,'-I','-S','-B','/opt/tileops-all-pairs/control/private_python.py',str(P/'runtime_entry.py'),cid,str(root/'checker.json')]
cmd=['npu-guard','run','--owner',env['GDN_CASE_OWNER'],'--wait','600','--timeout',('600' if cid.startswith('prefill-') else '300'),'--',*entry]
(root/'COMMAND.json').write_text(json.dumps({'argv':cmd,'env':{k:env[k] for k in pin['environment']},'case_id':cid},indent=2))
r=subprocess.run(cmd,cwd=root,env=env)
(root/'EXIT.json').write_text(json.dumps({'returncode':r.returncode}));sys.exit(r.returncode)
