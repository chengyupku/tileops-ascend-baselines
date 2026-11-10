import pathlib,sys,json,runpy,time,gc,contextlib,os
p=pathlib.Path(__file__).resolve().parent;label=sys.argv[1];rows=json.loads((p/(label+'.json')).read_text());start=time.monotonic();done=[]
for row in rows:
 if time.monotonic()-start>220:break
 op,cid=row;out=p/'runs'/(label+'--'+op+'--'+cid);out.mkdir(parents=True,exist_ok=True)
 (p/(label+'_progress.json')).write_text(json.dumps(dict(current=row,done=done)))
 print('CASE_START',row,flush=True)
 sys.argv=[str(p/'run_case.py'),op,cid,label]
 try:
  with (out/'console.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
   try:runpy.run_path(str(p/'run_case.py'),run_name='__main__')
   except SystemExit as e:
    if e.code not in (None,0):raise RuntimeError('checker exit '+str(e.code))
  d=json.loads((out/'checker.json').read_text());done.append(dict(key=row,status=d['status']))
  print('CASE_DONE',row,d['status'],flush=True)
  if d['status']!='verified':break
 except Exception as e:
  print('CASE_EXCEPTION',row,repr(e),flush=True);break
 finally:
  import torch
  torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
(p/(label+'_progress.json')).write_text(json.dumps(dict(current=None,done=done)))
