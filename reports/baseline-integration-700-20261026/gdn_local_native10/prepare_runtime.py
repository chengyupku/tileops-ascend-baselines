from pathlib import Path
import json,hashlib,shlex,shutil
W=Path(__file__).resolve().parent;ROOT=Path('/root/dyq/workspace');SDK=Path('/usr/local/Ascend/cann-9.2.0-beta.2');V=W/'install/vendors/kdapairs_transformer';P=W/'chunk_package';owner='baseline700-20261026/gdn_local_native10'
assert (V/'op_api/lib/libcust_opapi.so').is_file()
recipe={'dependency_closure_ready':True,'runtime_success_proven':False,'opapi':str(V/'op_api/lib/libcust_opapi.so'),'nnopbase':str(SDK/'lib64/libnnopbase.so'),'vendor':str(V),'checker':str(ROOT/'TileOPs/scripts/ascend_pair_check.py'),'required_environment':{'ASCEND_HOME_PATH':str(SDK),'ASCEND_TOOLKIT_HOME':str(SDK),'ASCEND_OPP_PATH':str(SDK/'opp'),'ASCEND_CUSTOM_OPP_PATH':str(V),'PYTHONDONTWRITEBYTECODE':'1','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','GDN_CASE_OWNER':owner},'python_paths':[str(ROOT/'.venv/lib/python3.12/site-packages'),str(SDK/'python/site-packages'),str(SDK/'opp/built-in/op_impl/ai_core/tbe')]}
(P/'RUNTIME_RECIPE.json').write_text(json.dumps(recipe,indent=2))
files=[Path(x['path']) for x in json.loads((W/'CANONICAL_SOURCE_PINS.json').read_text())]
files += [p for p in V.rglob('*') if p.is_file()]
files += [SDK/'lib64'/n for n in ['libnnopbase.so','libopapi_math.so','libascendcl.so','libruntime.so','libprofapi.so','libascend_dump.so']]
files += [P/n for n in ['binding.py','direct_acl.py','entry.py','resource_gate.py','RUNTIME_RECIPE.json']]
files += [W/n for n in ['case_launcher.py','CONTRACTS.json','CHUNK_CASES.json','CASE_RESOURCE_ESTIMATES.json']]
(P/'RUNTIME_PINS.json').write_text(json.dumps({'files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},indent=2))
cases=[x['case_id'] for x in json.loads((W/'SCOPE.json').read_text())];est=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text());cases.sort(key=lambda c:est[c]['estimated_device_budget_bytes']);assert cases[0]=='q35-35b-16x8192-bfloat16'
(W/'EXECUTION_ORDER.json').write_text(json.dumps(cases,indent=2))
for case in cases:
 d=W/case;d.mkdir(exist_ok=True);lines=['#!/bin/bash','set -eo pipefail',f'cd {W}','export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2',f'export GDN_RUN_DIR={d}']
 env=dict(recipe['required_environment']);env['PYTHONPATH']=':'.join(recipe['python_paths']);env['LD_LIBRARY_PATH']=':'.join(map(str,[SDK/'lib64',V/'op_api/lib',Path('/usr/local/Ascend/driver/lib64/driver'),Path('/usr/local/Ascend/driver/lib64/common'),ROOT/'.venv/lib/python3.12/site-packages/torch/lib',ROOT/'.venv/lib/python3.12/site-packages/torch_npu/lib']))
 env.update(FLA_NPU_DISABLE_PTH='1',PYTHONNOUSERSITE='1',ASCEND_GLOBAL_LOG_LEVEL='0',ASCEND_SLOG_PRINT_TO_STDOUT='0')
 for name in ['TMPDIR','PIP_CACHE_DIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','ASCEND_PROCESS_LOG_PATH','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','CCACHE_DIR']:
  (d/name).mkdir(exist_ok=True);env[name]=str(d/name)
 lines += ['export '+k+'='+shlex.quote(v) for k,v in env.items()]
 # Trace actual opened native metadata/device binaries without replacing code.
 lines.append(f'exec strace -f -qq -e trace=openat -o {d}/native-open.trace {ROOT}/.venv/bin/python -S -B {W}/case_launcher.py --op GDNFwdOp --case-id {case} --library ops-transformer --output {d}/checker.json')
 (d/'child.sh').write_text('\n'.join(lines)+'\n')
print(cases)
