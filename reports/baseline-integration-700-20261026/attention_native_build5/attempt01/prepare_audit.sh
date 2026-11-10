S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/source
grep -nE 'make|cmake|nproc|cpu|thread|python|\.so' "$S/cmake/scripts/prepare.sh" | tail -40
tail -25 "$S/build/prepare_build.log" 2>/dev/null
find "$S/build" -maxdepth 3 -name '*log*' | head -15
tail -15 /root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/native_build.log
