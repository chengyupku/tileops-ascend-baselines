set -e
S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
cd "$S/build"
/usr/bin/make -j1 -f CMakeFiles/merge_aicpu_json.dir/build.make merge_aicpu_json
sha256sum libtransformer_aicpu_kernels.so cust_aicpu_kernel.json
cat cust_aicpu_kernel.json
