from pathlib import Path
import json,hashlib,shutil,ast
T=Path(__file__).resolve().parent; D=T.parent/'bf16_local_gemm18'; W=Path('/root/dyq/workspace'); PRE=T.parent/'bf16_bmm_conv_grouped_prep10'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
release=json.loads((T.parent/'DEPLOYMENT_RELEASE_BF16_V5_20261026T2333.json').read_text())
for k in ['binding','native_host','device_object','original_generated_recipe']:
 v=release[k]; assert sha(Path(v['path']))==v['sha256'],k
shutil.copyfile(PRE/'adapter.py',T/'adapter.py');shutil.copyfile(D/'resource_gate.py',T/'resource_gate.py')
rows=json.loads((PRE/'CONTRACTS.json').read_text())['rows']
for r in rows:
 p=r['manifest_params'];op=r['key'][1]
 if op=='BmmFwdOp': e=p['B']*(p['M']*p['K']+p['K']*p['N']+p['M']*p['N']); cost=p['B']*p['M']*p['N']*p['K']
 elif op=='GroupedGemmFwdOp':e=p['M']*p['K']+p['G']*p['N']*p['K']+p['M']*p['N']+4*p['G'];cost=p['M']*p['N']*p['K']
 else:
  l=(p['L_in']+2*p['padding']-p['dilation']*(p['kW']-1)-1)//p['stride']+1
  e=p['N']*p['C_in']*p['L_in']+p['C_out']*p['C_in']*p['kW']+p['C_out']+p['N']*l*p['C_in']*p['kW']+p['N']*l*p['C_out'];cost=p['N']*l*p['C_out']*p['C_in']*p['kW']
 r['envelope']={'elements_including_full_batch_weights_outputs_packing':e,'host_required':2*1024**3+e*40,'device_required':2*1024**3+e*32,'disk_required':1024**3+e*8,'cost':cost}
(T/'CASES.json').write_text(json.dumps(rows,indent=2))
# Freeze the previously approved transitive runtime plus original sources; verify before reuse.
old=json.loads((D/'square-1k-bfloat16/FROZEN.json').read_text())['sha256']
pins={p:h for p,h in old.items() if '/install/' in p or '/site-packages/' in p or '/TileOPs/' in p or p.endswith('/tools/baseline_protocol.py') or p.endswith('/binding.py') or p=='/usr/local/bin/npu-guard'}
pins.update(json.loads((D/'SOURCE_PINS.json').read_text()))
for p,h in pins.items(): assert sha(Path(p))==h,p
for name in ['DEPLOYMENT_MANIFEST.json','SOURCE_PINS.json','ORIGINAL_BOUNDARY.json','DEVICE_COMMAND.json']:pins[str(D/name)]=sha(D/name)
for p in [T/'adapter.py',T/'capture.py',T/'resource_gate.py',T/'CASES.json',PRE/'adapter.py']:pins[str(p)]=sha(p)
(T/'FROZEN.json').write_text(json.dumps({'sha256':pins,'release':release},indent=2))
for r in rows:
 P=T/r['key'][2];P.mkdir(exist_ok=True)
 s='#!/bin/bash\nset -e\nsource /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh\n'
 s+='export PYTHONDONTWRITEBYTECODE=1\nexport ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2\nexport ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH ASCEND_AICPU_PATH=$ASCEND_HOME_PATH ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp\n'
 V=D/'install/vendors/bf16gmm_transformer'
 s+=f'export ASCEND_CUSTOM_OPP_PATH={V}\nexport ASCEND_CLASSIC_TRANSFORMER_VENDOR={V}\nexport LD_LIBRARY_PATH={V}/op_api/lib:$LD_LIBRARY_PATH\nexport ASCEND_GLOBAL_LOG_LEVEL=0\nexport OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
 for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR','CCACHE_DIR']:
  (P/name).mkdir(exist_ok=True);s+=f'export {name}={P/name}\n'
 s+=f'cd {W}\nexec {W}/.venv/bin/python {T}/capture.py {P} {r["key"][2]}\n';(P/'run.sh').write_text(s)
(T/'PREPARATION.json').write_text(json.dumps({'cases':len(rows),'frozen_pins_verified':len(pins),'adapter_identical':sha(T/'adapter.py')==sha(PRE/'adapter.py'),'capacity_formula':'2GiB margin + 40*all elements host; 32*all elements HBM includes BF16 originals/checker clones, FP32 widen and native outputs; 1GiB + 8*elements disk'},indent=2))
print('prepared',len(rows),'cases;',len(pins),'pins verified')
