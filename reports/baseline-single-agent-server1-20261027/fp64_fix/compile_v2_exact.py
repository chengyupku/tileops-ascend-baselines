from pathlib import Path
import subprocess,json,os,time,shutil,hashlib
P=Path(__file__).resolve().parent
b=P/'original-build/flash-linear-attention-npu/build'
guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert guard['child']['pid']==os.getpgrp() and guard['phase']=='running'
gen=b/'binary/ascend950/gen'
for name in ['compile_pids.txt','compile_stop.flag']:
 p=gen/name
 if p.exists():p.rename(p.with_name(name+'.before_v2_exact'))
(gen/'compile_pids.txt').write_text('')
param=json.loads((gen/'ChunkKdaFwdPrepare_9a5c7ba661140e04d3bf3c972e6629a6_param.json').read_text())
assert {x['name']:x['dtype'] for x in param['op_list'][0]['inputs']}['g']=='float32'
assert {x['name']:x['dtype'] for x in param['op_list'][0]['inputs']}['beta']=='float32'
targets=['chunk_kda_fwd_prepare_ascend950_3','chunk_kda_fwd_finalize_ascend950_0']
subprocess.run(['cmake','--build',str(b),'--target',*targets,'-j4'],check=True)
# Only original generated binary objects/configs, with original vendor packaging tool.
v=P/'fla_package/fla_npu/opp/vendors/fla_npu_transformer';kernel=v/'op_impl/ai_core/tbe/kernel';(kernel/'ascend950').mkdir(parents=True,exist_ok=True)
for stem in ['chunk_kda_fwd_prepare','chunk_fwd_h','chunk_kda_fwd_finalize']:
 src=b/'binary/ascend950/bin'/stem;assert list(src.glob('*.o')) and list(src.glob('*.json')),stem
 shutil.copytree(src,kernel/'ascend950'/stem,dirs_exist_ok=True)
script=P/'original-build/flash-linear-attention-npu/cmake/scripts/util/ascendc_ops_config.py'
subprocess.run(['python3',str(script),'-p',str(kernel/'ascend950'),'-s','ascend950','-o',str(kernel/'config/ascend950')],check=True)
(P/'FLA_V2_BINARY_DEPLOYMENT.json').write_text(json.dumps({'at':time.time(),'targets':targets,'unchanged_complete_api_library':str(v/'op_api/lib/libcust_opapi.so'),'binary_source':'original compiler outputs; original ascendc_ops_config.py metadata generation','files':{str(f.relative_to(v)):hashlib.sha256(f.read_bytes()).hexdigest() for f in kernel.rglob('*') if f.is_file()}},indent=2))
print('V2 original native components ready',flush=True)
