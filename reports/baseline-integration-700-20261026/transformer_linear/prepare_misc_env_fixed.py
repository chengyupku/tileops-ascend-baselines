from pathlib import Path
import json,subprocess,hashlib,datetime,os
W=Path(__file__).resolve().parent;R=W.parents[2];old=R/'.cache/all-pairs/resume22_server2_transformer_dense_state03';build=old/'native_build/state_passing/v6_build';p=json.loads((build/'BUILD_PLAN.json').read_text());source=old/'private_source/workspace_v6/ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state/torch_interface.cpp';d=W/'ssd_compile_env_fixed';d.mkdir(exist_ok=True)
cmd=[str(source) if x==p['original_source'] else x.replace(str(build/'dependencies.d'),str(d/'dependencies.d')).replace(str(build/'liboriginal_state_passing.so'),str(d/'liboriginal_chunk_state.so')) for x in p['command']]
rec=dict(argv=cmd,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_only=True,kernel_modified=False)
with (d/'compile.log').open('w') as f:
 try:rec['returncode']=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,ASCEND_HOME_PATH="/usr/local/Ascend/cann-9.1.0",ASCEND_OPP_PATH="/usr/local/Ascend/cann-9.1.0/opp",LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver",TMPDIR=str(d))).returncode
 except subprocess.TimeoutExpired:rec['returncode']=124
rec['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(d/'RECEIPT.json').write_text(json.dumps(rec,indent=2));print('SSD compile',rec['returncode'],flush=True)
