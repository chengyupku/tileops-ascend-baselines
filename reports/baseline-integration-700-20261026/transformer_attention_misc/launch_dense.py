import os,sys,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parents[2];cid=sys.argv[1];out=P/'runs'/cid;out.mkdir(parents=True,exist_ok=True)
cfg=json.load(open(W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json'))['groups']['attention'];env=os.environ.copy();env.update(cfg['environment']);env['LD_LIBRARY_PATH']=cfg['ld_library_path_prefix']
for n in ['TMPDIR','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','ASCEND_PROCESS_LOG_PATH']:
 p=P/'cache'/n;p.mkdir(parents=True,exist_ok=True);env[n]=str(p)
cmd=['npu-guard','run','--owner','baseline700-20261026/transformer_attention_misc','--wait','3600','--timeout','300','--',cfg['executable'],str(P/('run_batch.py' if cid.endswith('.json') else 'run_dense.py')),cid,*sys.argv[2:]]
start=time.time();(out/'command.json').write_text(json.dumps(cmd))
with (out/'guard.log').open('w') as f:rc=subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
(out/'guard.json').write_text(json.dumps({'returncode':rc,'start':start,'end':time.time(),'elapsed':time.time()-start}));print(rc)
