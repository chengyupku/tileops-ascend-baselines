import pathlib,json,time,os,sys
P=pathlib.Path(__file__).resolve().parent;h=sys.argv[1];state=json.loads((P/h/'STATE.json').read_text());out=[]
for tid,row in state.items():
 if row['status']!='failed':continue
 f=pathlib.Path('/dev/shm/torch-baseline1135')/(tid+'.pt')
 if not f.exists():continue
 attempt=P/h/'runs'/tid/row['attempt'];assert (attempt/'prepare_failure.json').exists() or (attempt/'device_failure.json').exists()
 st=f.stat();identity=(st.st_dev,st.st_ino)
 for proc in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   for fd in (proc/'fd').iterdir():
    try:
     x=fd.stat();assert (x.st_dev,x.st_ino)!=identity,'payload still open'
    except FileNotFoundError:pass
  except PermissionError:raise RuntimeError('cannot check handles')
 f.unlink();out.append({'id':tid,'size':st.st_size,'inode':st.st_ino,'failure_evidence':str(attempt)})
r={'at':time.time(),'removed_failed_CPU_temporary_payloads':out,'free_bytes':os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize};(P/h/('HANDOFF_CLEANUP_'+str(time.time_ns())+'.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r))
