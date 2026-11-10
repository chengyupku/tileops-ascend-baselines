from pathlib import Path
import sys,json,runpy,time,os
P=Path(__file__).resolve().parent;label=sys.argv[1];indices=[int(x) for x in sys.argv[2:]];rows=json.loads((P/'SCOPE.json').read_text());start=time.monotonic()
os.environ['SGL_CAPTURE_ROOT']=str(P/'runs'/f'{label}-{indices[0]:02d}'/'ASCEND_PROCESS_LOG_PATH')
for i in indices:
 if time.monotonic()-start>220:break
 r=rows[i];out=P/'runs'/f'{label}-{i:02d}'
 sys.argv=[str(P/'run_case.py'),r['baseline'],r['operator'],r['case_id'],str(out)]
 try:runpy.run_path(str(P/'run_case.py'),run_name='__main__')
 except SystemExit as e:
  if e.code not in (None,0):
   print('BATCH_STOP_EXIT',i,e.code,flush=True);break
 d=json.loads((out/'checker.json').read_text())
 if d['status']!='verified':
  print('BATCH_STOP_FAILED',i,d.get('error'),flush=True);break
 print('BATCH_VERIFIED',i,flush=True)
