R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build
sed -n '45,120p' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_opc_tool/op_compilation.py
cat "$R/compile1.log"
find "$B/binary/ascend950" -name '*lock*' -o -name '*.o' -o -name '*done' -o -name 'compile_stop.flag'
npu-guard status --json > "$R/guard_after.json"
cat "$R/guard_after.json"
