cat /usr/local/Ascend/cann-9.1.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/makeself.cmake
S=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source
head -60 "$S/build/scripts/install.sh"
grep -n -E 'python|npu-smi|upgrade|install-path|set_env|/usr/local' "$S/build/scripts/upgrade.sh" | head -35
