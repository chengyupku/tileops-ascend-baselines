O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import zipfile,pathlib
p=next(pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt02/stage/source/build_out').glob('*.whl'))
with zipfile.ZipFile(p) as z:
 for n in ['cann_ops_transformer/__init__.py','cann_ops_transformer/ops/__init__.py','cann_ops_transformer/utils.py']:
  if n in z.namelist():print(n,z.read(n).decode())
 print([n for n in z.namelist() if 'graph' in n or 'jit' in n or 'utils' in n][:20])
PY
head -55 "$O/attempt03/stage/source/cmake/scripts/util/ascendc_ops_config.py"
find "$O/attempt01/stage/wheels" -maxdepth 1 -type f -printf '%f\n'
tail -5 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/compile1.log
