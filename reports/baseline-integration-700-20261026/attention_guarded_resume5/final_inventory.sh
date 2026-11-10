/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import pathlib,json,hashlib,os
r=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5'); o=r.parent/'attention_native_build5'; b=o/'attempt03/stage/source/build'
sdk=pathlib.Path('/usr/local/Ascend/cann-9.1.0/python/site-packages/asc_opc_tool/op_compilation.py')
lines=sdk.read_text().splitlines(); print('\n'.join(lines[70:138]))
files=list((b/'binary/ascend950/gen').glob('kernel_meta_*/kernel_meta.lock'))
rows=[]
for p in files:
 text=p.read_text(); rows.append(dict(path=str(p),content=text,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recorded_pid_exists=pathlib.Path('/proc/'+text.strip()).exists()))
paths=[o/'attempt03/stage/bin/make',b/'CMakeFiles/flash_attn_ascend950_1.dir/build.make',b/'binary/ascend950/gen/FlashAttn-flash_attn-1.sh',sdk]
paths+=list((b/'op_api').rglob('*.so'))+list((b/'op_host').rglob('*.so'))+list((b/'custom').rglob('*.so'))+list((b/'binary/ascend950/bin').rglob('*'))+list((o/'attempt02/stage/source/build_out').glob('*.whl'))
art=[]
for p in dict.fromkeys(paths):
 if p.is_file(): art.append(dict(path=str(p),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
obj=dict(stale_locks=rows,artifacts=art,compiler_sdk_lock_logic='File existence + PID equality only; does not check whether recorded PID is alive',compile_stop_flag_exists=(b/'binary/ascend950/gen/compile_stop.flag').exists(),target1_done=(b/'binary/ascend950/gen/flash_attn_ascend950_1.done').exists())
(r/'FINAL_INVENTORY.json').write_text(json.dumps(obj,indent=2)); print(json.dumps(obj,indent=2))
PY
