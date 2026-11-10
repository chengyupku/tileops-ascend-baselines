import os,sys,json,pathlib,hashlib,runpy,time
P=pathlib.Path(__file__).resolve().parent; W=pathlib.Path('/opt/tileops-all-pairs/workspace')
op,cid,label=sys.argv[1:]; key=label+'--'+op+'--'+cid; out=P/'runs'/key;out.mkdir(parents=True,exist_ok=True)
meta=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());assert meta['phase']=='running' and meta['child']['pid']==os.getpgrp()
(out/'acquired.json').write_text(json.dumps({'utc_epoch':time.time(),'metadata':meta}))
os.chdir(out);os.environ['ASCEND_LAUNCH_BLOCKING']='1';sys.dont_write_bytecode=True
for name in ['TMPDIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH']:
 p=out/name;p.mkdir(exist_ok=True);os.environ[name]=str(p)
sys.path[:0]=[str(P),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')])
import integration700_sgl_linalg36 as adapter
import benchmarks.ascend_all_pairs.sgl as provider
assert (op,cid) in [(r['operator'],r['case_id']) for r in json.loads((P/'SCOPE.json').read_text())]
provider.PROVIDERS[op]=adapter.build
checker=W/'TileOPs/scripts/ascend_pair_check.py'
(out/'ROUTE.json').write_text(json.dumps({'factory':str(provider.PROVIDERS[op]),'adapter_hash':hashlib.sha256(pathlib.Path(adapter.__file__).read_bytes()).hexdigest(),'checker_hash':hashlib.sha256(checker.read_bytes()).hexdigest(),'interpreter':sys.executable,'source':str(adapter.SOURCE),'key':[op,cid]},indent=2))
sys.argv=[str(checker),'--op',op,'--case-id',cid,'--library','sgl-kernel-npu','--output',str(out/'checker.json')]
runpy.run_path(str(checker),run_name='__main__')
