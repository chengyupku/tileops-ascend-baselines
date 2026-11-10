import sys,os,json,time
from pathlib import Path
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'chunk_package'))
from entry import main
ids=sys.argv[1:]
for i in ids:
 os.environ['GDN_RUN_DIR']=str(W/i)
 sys.argv=['entry.py','--op','GDNFwdOp','--case-id',i,'--library','ops-transformer','--output',str(W/i/'checker.json')]
 t=time.time();(W/i/'DEVICE_STARTED.json').write_text(json.dumps({'timestamp':t}))
 try:main()
 except SystemExit as e:
  if e.code not in (None,0):raise
 c=json.loads((W/i/'checker.json').read_text())
 (W/i/'DEVICE_FINISHED.json').write_text(json.dumps({'timestamp':time.time()}))
 if c.get('status') not in ('verified','source_changed'):raise RuntimeError('Stop shared route: '+i)
