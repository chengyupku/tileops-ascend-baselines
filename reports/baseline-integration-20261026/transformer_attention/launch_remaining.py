from pathlib import Path
import sys,subprocess,json,os
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());keys=[r['case_id'] for r in scope if r['case_id']!='qwen3-30b-1k-float16'];keys.sort(key=lambda k:(not k.startswith('qwen'), k))
for i in range(0,len(keys),4):
 subprocess.run(['python3',str(P/'inspect_remote.py')],check=True,stdout=subprocess.DEVNULL)
 selected=keys[i:i+4];job=P/'batches'/str(i//4+1);job.mkdir(parents=True,exist_ok=False);(job/'cases.json').write_text(json.dumps(selected))
 cmd=['npu-guard','run','--owner','selected-integration-20261026/transformer_attention','--wait','3600','--timeout','300','--','/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_batch.py'),str(job)]
 (job/'COMMAND.json').write_text(json.dumps(cmd,indent=2));env=os.environ.copy()
 for var in ['TMPDIR','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR']:
  target=job/var;target.mkdir();env[var]=str(target)
 env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
 with (job/'guard.log').open('x') as f:rc=subprocess.run(cmd,cwd=job,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 (job/'GUARD_RESULT.json').write_text(json.dumps({'returncode':rc}));print(json.dumps({'batch':i//4+1,'cases':selected,'guard_exit_code':rc}),flush=True)
 if rc:sys.exit(rc)
