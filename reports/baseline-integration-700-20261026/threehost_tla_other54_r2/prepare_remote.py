import pathlib,shutil,json,hashlib,time
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/threehost_tla_other54_r2';P.mkdir(parents=True,exist_ok=True)
records=[]
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);records.append(dict(original=str(src),private=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
for root,dest in [(W/'TileOPs/benchmarks',P/'frozen/benchmarks'),(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/private_tla',P/'private_tla')]:
 for f in root.rglob('*.py'):cp(f,dest/f.relative_to(root))
cp(W/'TileOPs/scripts/ascend_pair_check.py',P/'checker/ascend_pair_check.py')
cp(W/'reports/baseline-integration-700-20261026/tla_simple_tilelang/state_adapter_pto.py',P/'state_adapter_pto.py')
(P/'FREEZE.json').write_text(json.dumps(records,indent=2))
rows=[]
for sdk in [pathlib.Path('/usr/local/Ascend/cann-9.1.0'),*list((W/'.cache/all-pairs/remote_native_deepgemm_deepep').glob('sdk-*'))]:
 for rel in ['asc/include/c_api/sys_var/sys_var.h','asc/include/c_api/cube_datamove/cube_datamove.h','asc/include/c_api/utils/enum.h']:
  f=sdk/rel
  if f.exists():
   t=f.read_text(); rows.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),symbols=[l for l in t.splitlines() if any(s in l for s in ['asc_set_gm2l1_nz_para','asc_set_l0c_copy_nz_para','asc_quant_mode','asc_unit_flag_mode'])]))
(P/'SDK_SIGNATURES.json').write_text(json.dumps(rows,indent=2));print(json.dumps(dict(frozen=len(records),sdk_headers=rows),indent=2))
