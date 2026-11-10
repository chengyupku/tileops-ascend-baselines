import pathlib,json,os,subprocess,time,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent; E=P/'cpu_attempt02';E.mkdir(exist_ok=False)
d=json.loads((P/'cpu_attempt01/sdk92_driver.json').read_text()); cmd=d['argv'];cmd.remove('-###');cmd[-1]=str(E/'original_auto.so')
sdk=pathlib.Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2')
env=os.environ.copy();env.update(ASCEND_HOME_PATH=str(sdk),ASCEND_HOME=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']),TMPDIR=str(E))
t=time.time()
try:
 r=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=E,timeout=120);rc=r.returncode;log=r.stdout+r.stderr
except subprocess.TimeoutExpired as e: rc=124;log=str(e)
(E/'compile.log').write_text(log)
(E/'compile.json').write_text(json.dumps(dict(argv=cmd,env_overrides={k:env[k] for k in ['ASCEND_HOME_PATH','ASCEND_HOME','ASCEND_TOOLKIT_HOME','CPATH','TMPDIR']},returncode=rc,wall_seconds=time.time()-t,device_access=False,source_sha256=hashlib.sha256((P/'cpu_attempt01/original_auto.cpp').read_bytes()).hexdigest(),note='CPU-only compiler invocation of preserved prior original auto input; no generated source or architecture alteration; no runtime loading; not numerical acceptance'),indent=2))
print(rc,log[-7000:])
