S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/source
sed -n '10,24p' "$S/cmake/scripts/prepare.sh"
sed -n '125,165p' "$S/cmake/scripts/prepare.sh"
sed -n '1,145p' "$S/cmake/third_party/ops-tensor.cmake"
grep -n 'JOB_NUM\|CPU_NUM' "$S/cmake/scripts/prepare.sh"
