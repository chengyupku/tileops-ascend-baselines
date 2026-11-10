S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
cat "$S/attention/flash_attn_metadata/CMakeLists.txt"
cat "$S/build/CMakeFiles/cust_aicpu_kernels.dir/build.make" | grep -E '^\s.*(python|cmake|gcc|g\+\+|aicpu)' | head -25
ls -l "$S/build/libtransformer_aicpu_kernels.so"
grep -n 'flash_attn_metadata' "$S/build/cust_aicpu_kernel.json" "$S/build/CMakeFiles/cust_aicpu_kernels.dir/DependInfo.cmake" 2>/dev/null || true
cat "$S/cmake/scripts/util/gen_version_info.sh"
