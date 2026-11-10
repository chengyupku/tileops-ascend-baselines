import pathlib,json,subprocess,hashlib,shutil,os,time,shlex
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/tla_sdk_recovery36';P.mkdir(exist_ok=True,parents=True)
old=W/'reports/baseline-integration-700-20261026/tla_simple_tilelang'
sdk=W/'.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
for f in sdk.glob('**/*version*'):
 if f.is_file() and len(f.relative_to(sdk).parts)<5 and f.stat().st_size<15000:
  print('VERSION',str(f),f.read_text(errors='replace')[:1200])
checks=list((old/'runs').glob('tilelang_*/checker.json'))
for f in checks:
 d=json.loads(f.read_text());err=d.get('error','')
 if 'Command: ' not in err:continue
 cmd=shlex.split(err.split('Command: ',1)[1].split('\n',1)[0])
 src=pathlib.Path(next(x for x in cmd if x.endswith('.asc')))
 print('CHECKER',f,'SOURCE',src,'EXISTS',src.exists())
 if not src.exists():
  if '\nSource:\n' not in err: continue
  src=P/'recovered_original_fp8.asc';src.write_text(err.split('\nSource:\n',1)[1])
  cmd=[str(src) if x.endswith('.asc') else x for x in cmd]
 dest=P/(f.parent.name+'.asc');shutil.copyfile(src,dest)
 cmd=[str(sdk/'tools/bisheng_compiler/bin/bisheng') if i==0 else str(dest) if x==str(src) else str(P/(f.parent.name+'.o')) if i and cmd[i-1]=='-o' else x for i,x in enumerate(cmd)]
 env=os.environ.copy();env.update(ASCEND_HOME_PATH=str(sdk),ASCEND_TOOLKIT_HOME=str(sdk),ASCEND_HOME=str(sdk),CPATH=':'.join(str(sdk/x) for x in ['asc/include','asc','include']))
 start=time.time()
 r=subprocess.run(cmd,env=env,cwd=P,capture_output=True,text=True,timeout=120)
 (P/(f.parent.name+'.compile.log')).write_text(r.stdout+r.stderr)
 record=dict(command=cmd,source_original=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),returncode=r.returncode,seconds=time.time()-start,log=f.parent.name+'.compile.log')
 (P/(f.parent.name+'.compile.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record));print((r.stdout+r.stderr)[-7000:])
 break
for rel in ['asc/include/c_api/reg_compute/reg_vector.h','asc/include/c_api/reg_compute/reg_store.h','asc/include/c_api/utils/enum.h','tools/bisheng_compiler/bin/bisheng']:
 f=sdk/rel;print('HASH',str(f),hashlib.sha256(f.read_bytes()).hexdigest())
