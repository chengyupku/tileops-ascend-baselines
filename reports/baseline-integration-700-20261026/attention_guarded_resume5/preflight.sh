set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03
mkdir -p "$R"
cat "$B/stage/bin/make"
sha256sum "$B/stage/bin/make" "$B/stage/source/build/CMakeFiles/flash_attn_ascend950_1.dir/build.make" "$B/stage/source/build/binary/ascend950/gen/FlashAttn-flash_attn-1.sh"
cat "$B/stage/source/build/binary/ascend950/gen/FlashAttn-flash_attn-1.sh"
ls -l "$B/stage/source/build/binary/ascend950/gen/"
cat "$B/stage/source/build/CMakeFiles/flash_attn_ascend950_1.dir/build.make"
npu-guard run --help
