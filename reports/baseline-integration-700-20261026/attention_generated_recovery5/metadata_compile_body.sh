set -e
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build
cd "$B"
/usr/bin/make -j1 -f attention/flash_attn_metadata/op_kernel_aicpu/CMakeFiles/flash_attn_metadata_cust_obj.dir/build.make attention/flash_attn_metadata/op_kernel_aicpu/CMakeFiles/flash_attn_metadata_cust_obj.dir/build
/usr/bin/make -j1 -f CMakeFiles/cust_aicpu_kernels.dir/build.make cust_aicpu_kernels
