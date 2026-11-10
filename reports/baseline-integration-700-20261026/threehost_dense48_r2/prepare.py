import pathlib,subprocess,json,datetime
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_dense48_r2'
def remote(host,cmd,name):
 t=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run([SSH,host,cmd],text=True,capture_output=True);(P/name).write_text(json.dumps(dict(time=t,cmd=cmd,rc=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2));return p
hosts={'local':None,'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}
for h,a in hosts.items():
 if a:remote(a,'npu-guard status --json',h+'_status.json')
 else:
  p=subprocess.run(['npu-guard','status','--json'],text=True,capture_output=True);(P/'local_status.json').write_text(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)))
cmd=f'''mkdir -p {R}; cp -a /opt/tileops-all-pairs/workspace/TileOPs {R}/TileOPs; cp -a /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider {R}/vendor; du -sh {R}/TileOPs {R}/vendor; /opt/tileops-ci/runtime/bin/python -I -S -B /opt/tileops-all-pairs/control/private_python.py -c 'import sys; print(sys.version)' '''
r=remote(hosts['server1'],cmd,'prepare_remote.json');print(r.returncode,r.stdout,r.stderr)
(P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='private_snapshot_prepared',currentkey='ops-transformer/GemmFP8FwdOp/ds-v3-m1-down-blk-float8_e4m3fn-bfloat16',host='server1',lastprogress=datetime.datetime.now(datetime.timezone.utc).isoformat(),blockedreason=None)))
