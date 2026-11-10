set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
B="$O/attempt03/stage/source/build"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$O/attempt03/stage/bin:$PATH"
export PYTHONPATH="$R/python:$O/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages"
cd "$B"
test -f binary/ascend950/gen/flash_attn_ascend950_0.done
test -f binary/ascend950/gen/flash_attn_ascend950_1.done
test -f libtransformer_aicpu_kernels.so
/usr/bin/make -j1 -f CMakeFiles/ops_config_ascend950.dir/build.make ops_config_ascend950
cmake --install . --prefix "$R/package_stage" > "$R/cmake_install.log" 2>&1
cd "$R/package_stage"
M=/usr/local/Ascend/cann-9.1.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules
bash "$M/util/makeself/makeself.sh" --header "$M/util/makeself/makeself-header.sh" --help-header ./help.info --tar-format posix --gzip --complevel 4 --nomd5 --sha256 ./ "$R/cann-ops-transformer-attention_native_build5_linux-aarch64.run" "version:1.0" ./install.sh > "$R/makeself.log" 2>&1
bash "$R/cann-ops-transformer-attention_native_build5_linux-aarch64.run" --noexec --target "$R/install_payload" > "$R/extract_package.log" 2>&1
sha256sum "$R/cann-ops-transformer-attention_native_build5_linux-aarch64.run"
