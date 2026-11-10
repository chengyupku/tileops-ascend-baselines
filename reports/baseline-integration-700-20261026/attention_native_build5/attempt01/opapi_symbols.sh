R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
S="$R/attempt03/stage/source/build"
readelf -h "$S/libcust_opapi.so" | head -18
nm -D "$S/libcust_opapi.so" | grep -E 'aclnnFlashAttn|FlashAttnMetadata'
find "$S/binary/ascend950/bin/flash_attn" -type f | head -10
find "$S/binary/ascend950/gen" -name '*.o' -printf '%f\n' | head -10
git -C "$R/attempt03/stage/source/third_party/ops-tensor" rev-parse HEAD
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
