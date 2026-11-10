from pathlib import Path
import json,hashlib,re,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in json.loads((P/'SCOPE.json').read_text()):
 D=P/row['case_id']; receipt=D/'result/representative.json'
 if not receipt.exists():continue
 records=[];logs=[];artifacts=[]
 for f in (D/'runtime').rglob('*'):
  if not f.is_file():continue
  if 'ASCEND_PROCESS_LOG_PATH' in str(f):
   lines=f.read_text(errors='replace').splitlines();matches=[x for x in lines if any(t in x for t in ['Register Kernel succ','BinaryLaunch','binaryLaunch','StreamLaunchKernel','GroupedMatmul_','aclnnGroupedMatmulV5','GetWorkspaceSize'])]
   if matches:logs.append({'path':str(f),'sha256':sha(f),'size':f.stat().st_size,'lines':matches})
  elif f.suffix in ['.o','.json','.so','.cpp','.py']:
   artifacts.append({'path':str(f),'sha256':sha(f),'size':f.stat().st_size})
 events=json.loads((D/'result/NATIVE_CALLS.json').read_text()) if (D/'result/NATIVE_CALLS.json').exists() else []
 data={'case':row['case_id'],'time':time.time(),'native_events':len(events),'actual_host_apis':sorted(set((e['function'],e.get('path'),e.get('sha256')) for e in events if e['event']=='enter')),'driver_evidence':logs,'runtime_artifacts':artifacts,'receipt_sha256':sha(receipt)}
 (D/'ACTUAL_NATIVE.json').write_text(json.dumps(data,indent=2))
 print(row['case_id'],len(logs),len(artifacts))
