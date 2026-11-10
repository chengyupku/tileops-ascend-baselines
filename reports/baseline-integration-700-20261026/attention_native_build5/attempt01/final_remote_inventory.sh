R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
npu-guard status --json > "$R/guard_after_stop.json"
cat "$R/guard_after_stop.json"
ps -eo pid,pgid,args | awk '$2==2676912 || $2==2668466'
python3 - "$R" <<'PY'
import pathlib,json,hashlib,subprocess,sys,zipfile
r=pathlib.Path(sys.argv[1]); result={}
for a in ['attempt01','attempt02','attempt03']:
 root=r/a/'stage/source';items=[]
 for glob in ['build_out/*.run','build_out/*.whl','build/*.so','build/binary/ascend950/bin/flash_attn/*']:
  for p in root.glob(glob):
   if p.is_file():items.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 result[a]=dict(artifacts=items,receipt=json.loads((r/a/'native_build_receipt.json').read_text()))
w=r/'attempt02/stage/source/build_out/cann_ops_transformer-1.0.0-py3-none-any.whl'
with zipfile.ZipFile(w) as z:
 result['wrapper_wheel']=dict(path=str(w),members=len(z.namelist()),flash_attn_members=[n for n in z.namelist() if 'flash_attn' in n],native_so_members=[n for n in z.namelist() if n.endswith('.so')])
result['native_launch']=dict(executed=False,loaded_opapi=None,loaded_kernel=None,reason='No deployment or checker; build was stopped on unexpected compiler driver handle')
(r/'ARTIFACT_INVENTORY.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
PY
S="$R/attempt03/stage/source/build"
for f in "$S/libcust_opmaster_rt2.0.so" "$S/op_api/libcust_opapi.so"; do if test -f "$f"; then file "$f"; readelf -h "$f" | head -18; nm -D "$f" | grep -E 'FlashAttn|flash_attn' | head -10; fi; done
