"""Foreground serial subprocesses within one bounded host-guard reservation."""
import pathlib,sys,json,subprocess,time
P=pathlib.Path(__file__).resolve().parent
manifest=pathlib.Path(sys.argv[1]);jobs=json.loads(manifest.read_text());results=[]
for j in jobs:
 index=j['index'];variant=j['variant'];attempt=j['attempt'];mode=j.get('mode','pilot')
 label=f'{mode}-{index}-{variant}-{attempt}'
 if (P/'DEVICE_STOP').exists():break
 result_path=P/'runs'/str(index)/variant/attempt/'result.json'
 if result_path.exists():
  results.append({'job':j,'status':'already_present_not_rerun'});continue
 with (P/f'{label}.log').open('w') as f:
  start=time.time()
  try:
   r=subprocess.run(['bash',str(P/'launch_v9.sh'),mode,str(index),variant,attempt],stdout=f,stderr=subprocess.STDOUT,timeout=300)
   rc=r.returncode
  except subprocess.TimeoutExpired:
   # Stop this reservation; guard will drain all descendants before releasing.
   results.append({'job':j,'exit_code':124,'seconds':time.time()-start});break
 results.append({'job':j,'exit_code':rc,'seconds':time.time()-start})
 (manifest.with_suffix('.progress.json')).write_text(json.dumps(results,indent=2))
 print('BATCH_DONE',label,rc,flush=True)
 # Device/driver exceptions must be reviewed before another case.
 if rc and result_path.exists():
  error=json.loads(result_path.read_text()).get('error','')
  if any(s in error for s in ('507014','507035','507011','Device synchronize failed','AI Core','memory admission')):break
(manifest.with_suffix('.progress.json')).write_text(json.dumps(results,indent=2))
if any(x.get('exit_code')==124 for x in results):sys.exit(124)
