from pathlib import Path
import shutil,json,hashlib,subprocess
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn_grouped3_build');P.mkdir(exist_ok=True)
B=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group3/recovery_1155/tasks/opsnn-host-build')
O=P.parent/'threehost_nn11'
shutil.copytree(B/'source',P/'source',ignore=shutil.ignore_patterns('build','__pycache__'))
shutil.copytree(B/'asc_cmake',P/'asc_cmake')
# Restore byte-identical operator source closure already verified by prior worker.
pins=json.loads((O/'ORIGINAL_MATMUL_PINS.json').read_text())
for rel in pins:
 src=O/'sources/ascend_baselines/ops-nn'/rel;dst=P/'source'/rel
 shutil.copy2(src,dst)
rel='matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp'
assert hashlib.sha256((P/'source'/rel).read_bytes()).hexdigest()==pins[rel]
changes=[]
for base in [P/'source/cmake',P/'asc_cmake']:
 for f in base.rglob('*.cmake'):
  s=f.read_text();n=s.replace(str(B),str(P))
  if s!=n:f.write_text(n);changes.append(str(f))
for name in ['tmp','cache','work','logs']: (P/name).mkdir(exist_ok=True)
(P/'env.sh').write_text((B/'env.sh').read_text().replace(str(B),str(P)))
s=(B/'configure.sh').read_text().replace(str(B),str(P)).replace('-DENABLE_BINARY=OFF','-DENABLE_BINARY=ON').replace('mat_mul_v3;batch_mat_mul_v3;foreach_mul_list','mat_mul_v3;batch_mat_mul_v3')
(P/'configure.sh').write_text(s)
for name,exe in [('guard_codegen.sh','/usr/local/Ascend/ascend-toolkit/latest/tools/opbuild/op_build'),('guard_es_codegen.sh','/usr/local/Ascend/ascend-toolkit/latest/bin/gen_esb')]:
 script='#!/bin/bash\nset -e\nexec npu-guard run --owner baseline700-20261026/threehost_nn_grouped3_build --wait 120 --timeout 600 -- '+exe+' "$@"\n'
 (P/name).write_text(script);(P/name).chmod(0o755)
(P/'PREPARE_ARM.json').write_text(json.dumps(dict(restored_original_source_files=len(pins),restored_infer_hash=pins[rel],source_audit_reused='threehost_nn_grouped3_native_repair/DIAGNOSIS.json',build_path_relocations=changes,original_target='ascendc_bin_ascend950_mat_mul_v3',full_OPP_copied=False),indent=2))
print('Prepared',len(pins),'original source files; no device initialized')
