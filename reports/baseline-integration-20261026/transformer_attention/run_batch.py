from pathlib import Path
import sys,json,runpy,gc
P=Path(__file__).resolve().parent;job=Path(sys.argv[1]);rows=json.loads((job/'cases.json').read_text())
for cid in rows:
 sys.argv=[str(P/'run_case.py'),cid,str(job/cid)]
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 except SystemExit as e:
  if e.code not in (None,0):raise
 d=json.loads((job/cid/'checker.json').read_text())
 assert d['status']=='verified' and d['verification']['checked_outputs']==d['verification']['total_outputs']==1
 print('CASE_VERIFIED '+cid,flush=True)
 gc.collect()
 import torch
 torch.npu.synchronize()
