S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/source
grep -nE 'RULE_LAUNCH|rule_launch|CUSTOM_OPTION=|export.*JOB|MAKEFLAGS' "$S/build.sh" | head -20
grep -R -nE 'protobuf_static_build|cmake --build|--parallel|-j[0-9]|nproc' "$S/cmake" "$S/third_party/cann-cmake/third_party/protobuf.cmake" | head -35
git -C "$S/third_party/ops-tensor" rev-parse HEAD
ps -eo pid,pgid,args | awk '$2==2668466'
