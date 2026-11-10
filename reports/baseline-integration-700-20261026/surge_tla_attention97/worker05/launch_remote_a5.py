from pathlib import Path
import sys,os,json,subprocess,time
P=Path(__file__).resolve().parent;op,cid=sys.argv[1:]
assert ['tilelang-ascend',op,cid] in [[r['baseline'],r['operator'],r['case_id']] for r in json.loads((P/'SCOPE.json').read_text())]
out=P/'runs'/(cid+'-legacy-pto-a5');out.mkdir(parents=True,exist_ok=True)
assert not (out/'guard.json').exists(),'No automatic repeat'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for n in ['TMPDIR','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH']:
 d=out/n;d.mkdir(exist_ok=True);env[n]=str(d)

cmd=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker05','--wait','120','--timeout','600','--','/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_remote_a5.py'),op,cid]
rec={'key':['tilelang-ascend',op,cid],'argv':cmd,'started':time.time()}
(out/'COMMAND.json').write_text(json.dumps(rec,indent=2))
with (out/'guard.log').open('x') as f:r=subprocess.run(cmd,env=env,cwd=out,stdout=f,stderr=subprocess.STDOUT)
rec.update(returncode=r.returncode,finished=time.time());(out/'guard.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec));sys.exit(r.returncode)
