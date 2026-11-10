import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent
for label in ['batch03','batch04']:
 out=P/label/'RETAINED_NATIVE.jsonl';files={};n=0
 with out.open('w') as stream:
  for log in (P/'runs').glob(label+'--*/output/ASCEND_PROCESS_LOG_PATH/**/*.log'):
   data=log.read_bytes();files[str(log.relative_to(P))]=hashlib.sha256(data).hexdigest()
   for i,line in enumerate(data.decode(errors='replace').splitlines(),1):
    if any(x in line for x in ['Register Kernel succ','StreamLaunchKernel','bin path is ']):
     stream.write(json.dumps({'file':str(log.relative_to(P)),'line':i,'text':line})+'\n');n+=1
 (P/label/'RETAINED_NATIVE_SUMMARY.json').write_text(json.dumps({'retained_only':True,'note':'CANN log root fixed at process startup; subsequent cases share first case log directory. These are actual retained events; rotations may omit older events. No rerun.','events':n,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_log_hashes':files},indent=2))
print('retained logs collected')
