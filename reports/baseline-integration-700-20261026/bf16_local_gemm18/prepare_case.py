from pathlib import Path
import json,hashlib,sys,shlex
W=Path('/root/dyq/workspace');T=Path(__file__).resolve().parent;P=T/sys.argv[1];P.mkdir(exist_ok=False)
V=T/'install/vendors/bf16gmm_transformer'
s=(T/'runtime-probe.sh').read_text().split('cd /root/dyq/workspace')[0]
s+=f'\ncd {W}\nexport ASCEND_CUSTOM_OPP_PATH={V}\nexport ASCEND_CLASSIC_TRANSFORMER_VENDOR={V}\nexport LD_LIBRARY_PATH={V}/op_api/lib:$LD_LIBRARY_PATH\nexport ASCEND_GLOBAL_LOG_LEVEL=0\nexport OMP_NUM_THREADS=4\nexport MKL_NUM_THREADS=4\n'
for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TMPDIR','TMP','TEMP','ACL_OP_COMPILER_CACHE_DIR']:
 (P/name).mkdir(exist_ok=True);s+=f'export {name}={P/name}\n'
s+=f'exec {W}/.venv/bin/python {T}/capture.py {P} {shlex.quote(sys.argv[1])}\n';(P/'run.sh').write_text(s)
files=[T/'capture.py',T/'binding.py',T/'resource_gate.py',P/'run.sh',W/'TileOPs/scripts/ascend_pair_check.py',W/'tools/baseline_protocol.py',Path('/usr/local/bin/npu-guard')]
files += [p for p in V.rglob('*') if p.is_file()]
for root in [W/'TileOPs/workloads',W/'TileOPs/benchmarks']:files+=list(root.rglob('*.py'))
# Runtime pins are shared read-only; per-case loaded maps are retained separately.
files += [p for root in [W/'.venv/lib/python3.12/site-packages/torch',W/'.venv/lib/python3.12/site-packages/torch_npu'] for p in root.rglob('*.so')]
h={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(P/'FROZEN.json').write_text(json.dumps({'sha256':h},indent=2));print('prepared',sys.argv[1],len(h))
