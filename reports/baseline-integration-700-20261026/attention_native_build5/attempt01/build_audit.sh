S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01/stage/source
sed -n '1,105p' "$S/cmake/scripts/util/const_var.py"
grep -R -nE 'torch_npu|import torch|aclInit|aclrtSetDevice|acl\.init|acl\.rt\.set_device' "$S/cmake" "$S/third_party/cann-cmake/scripts" "$S/torch_extension/setup.py" | head -30
grep -nE '^(import|from)|CDLL|subprocess|Popen|set_device|init\(' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/asc_op_compiler/{compile_op.py,ascendc_compile_base.py,get_op_tiling.py} | head -80
cat "$S/attention/flash_attn/CMakeLists.txt" "$S/attention/flash_attn_metadata/CMakeLists.txt"
find /usr/local/Ascend/cann-9.1.0 -maxdepth 4 -name version.info | head -25
ls /opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/share/cmake/Torch
cat /opt/tileops-ci/runtime/lib/python3.12/site-packages/torch/share/cmake/Torch/TorchConfig.cmake | grep -n -A3 ABI
