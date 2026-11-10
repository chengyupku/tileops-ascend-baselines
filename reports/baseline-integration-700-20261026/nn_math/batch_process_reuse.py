"""Explicit same-family short batch; imports/compiler cache retained."""
import contextlib,datetime,json,runpy,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;jobs=json.loads(Path(sys.argv[1]).read_text());assert len({tuple(j[:2]) for j in jobs})==1;started=time.monotonic()
for base,op,cid,attempt in jobs:
 if time.monotonic()-started>245:break
 out=P/attempt;t=time.time();rc=0;sys.argv=[str(P/'run_case.py'),base,op,cid,attempt]
 with (out/'case.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
  try:runpy.run_path(str(P/'run_case.py'),run_name='__main__')
  except SystemExit as e:rc=e.code or 0
  except BaseException:traceback.print_exc();rc=1
 (out/'CHILD_RESULT.json').write_text(json.dumps(dict(returncode=rc,elapsed_s=time.time()-t)));print(attempt,rc,flush=True)
 if rc:break
