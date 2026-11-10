set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
B="$O/attempt03"
mkdir -p "$R/tmp" "$R/cache" "$R/ascend_work"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$O/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$R/torch_extensions" MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_COMPILE_CORE_NUMBER=1
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/cache" CCACHE_DIR="$R/ccache" ASCEND_CACHE_PATH="$R/ascend_cache" ASCEND_WORK_PATH="$R/ascend_work"
export PATH="$B/stage/bin:$PATH"
cd "$B/stage/source/build"
set +e
npu-guard run --owner 'baseline700-20261026/attention_guarded_resume5' --wait 600 --timeout 600 -- "$B/stage/bin/make" -f CMakeFiles/flash_attn_ascend950_1.dir/build.make flash_attn_ascend950_1 VERBOSE=1 > "$R/compile1.log" 2>&1
rc=$?
cat /run/lock/ascend-npu/device-0.lock > "$R/compile1.guard.json"
printf '%s\n' "$rc" > "$R/compile1.exit"
tail -60 "$R/compile1.log"
cat "$R/compile1.guard.json"
exit "$rc"
