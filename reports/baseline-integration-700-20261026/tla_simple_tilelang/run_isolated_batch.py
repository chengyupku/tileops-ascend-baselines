from pathlib import Path
import sys,json,subprocess,time,os
P=Path(__file__).resolve().parent;batch=json.loads((P/sys.argv[1]).read_text());start_batch=time.time()
meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
assert meta['owner']=='baseline700-20261026/tla_simple_tilelang' and meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
for index,(op,cid) in enumerate(batch):
 if time.time()-start_batch>250:break
 one=P/('isolated_case_'+str(index)+'.json');one.write_text(json.dumps([[op,cid]]))
 out=P/'runs'/(op+'_'+cid+'_v7');out.mkdir(parents=True,exist_ok=True)
 assert not (out/'checker.json').exists()
 (out/'GUARD_RUNNING.json').write_text(json.dumps(meta,indent=2))
 start=time.time()
 rc=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_batch9.py'),one.name],cwd=P).returncode
 if not (out/'guard.json').exists():
  (out/'guard.json').write_text(json.dumps(dict(key=['tilelang-ascend',op,cid],started=start,finished=time.time(),returncode=128-rc if rc<0 else rc,shared_guard=True,process_isolated=True),indent=2))
 receipt=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else {}
 print(json.dumps(dict(isolated_key=[op,cid],child_rc=rc,status=receipt.get('status'))),flush=True)
 if receipt.get('sync_error'):break
