O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
B=$O/attempt03/stage/source/build
rg -n 'asc_opc|Generating|bash |python|cmake -P' "$B/CMakeFiles"/*/build.make | grep -E 'metadata|ops_config|ops_transformer_config|gen_version' | head -35
cat "$B/cmake_install.cmake" | tail -100
ls "$O/attempt01"; ls "$O/attempt01/stage"
/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import importlib.metadata as m
for n in ['torch','torch-npu','ml-dtypes','torchair']:
 try:
  d=m.distribution(n);print(n,d.version,d.requires)
 except m.PackageNotFoundError:pass
PY
