B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01
cat "$B/download.log"
/opt/tileops-ci/runtime/bin/python3 -m pip --version
python3 -m pip --version
ls /opt/tileops-ci/runtime/bin
cat /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base/{__init__.py,asc_op_compiler/__init__.py}
grep -R -nE 'torch_npu|aclInit|aclrtSetDevice|acl\.init|acl\.rt\.set_device|aclrtMalloc' /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base | head -25
