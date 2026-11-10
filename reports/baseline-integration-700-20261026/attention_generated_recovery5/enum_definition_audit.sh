head -45 /usr/local/Ascend/cann-9.1.0/include/acl/acl_base.h
grep -R -n 'ACL_HIFLOAT8\|ACL_HIFLOAT4' /usr/local/Ascend/cann-9.1.0/include/acl /usr/local/Ascend/cann-9.1.0/include/aclrt /opt/tileops-ci/runtime/lib/python3.12/site-packages/torch_npu/include/third_party/acl/inc --include='*.h' 2>/dev/null | head -15
