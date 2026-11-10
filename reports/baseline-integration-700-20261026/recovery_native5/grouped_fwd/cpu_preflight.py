from pathlib import Path
import subprocess,concurrent.futures,json,datetime,shlex,hashlib
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code='''import pathlib,json,platform,sys,sysconfig,shutil
W=pathlib.Path(WORKSPACE)
V=W/'.cache/ascend-classic/native-pilot-20261020/ops-nn/install/vendors/classicpilot_nn'
B=W/'reports/baseline-integration-700-20261026/threehost_nn_grouped3_build'
c=V/'op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'
config=json.loads(c.read_text()) if c.exists() else {}
print(json.dumps(dict(arch=platform.machine(),python=sys.version,abi=sysconfig.get_config_var('SOABI'),vendor=str(V),vendor_exists=V.exists(),config_exists=c.exists(),grouped_config_keys=[k for k in config if 'group' in k.lower()],matmul_config_keys=[k for k in config if 'matmul' in k.lower()],prepared_build=str(B),prepared_build_exists=(B/'build').exists(),metadata_exists=(B/'build/autogen').exists(),bisheng=shutil.which('bisheng'),opbuild=shutil.which('op_build'),source_matmul_dirs=sorted(x.name for x in (W/'ascend_baselines/ops-nn/matmul').iterdir()) if (W/'ascend_baselines/ops-nn/matmul').exists() else None)))
'''
def run(host):
 w='/root/dyq/workspace' if host=='local' else '/opt/tileops-all-pairs/workspace'
 script=code.replace('WORKSPACE',repr(w))
 cmd=['python3','-c',script] if host=='local' else [SSH,host,'python3 -c '+shlex.quote(script)]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 return dict(host=host,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
with concurrent.futures.ThreadPoolExecutor() as ex: rows=list(ex.map(run,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
(P/'CPU_PREFLIGHT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=rows),indent=2))
for r in rows:print(r['host'],r['rc'],r['stdout'][:1800],r['stderr'])
