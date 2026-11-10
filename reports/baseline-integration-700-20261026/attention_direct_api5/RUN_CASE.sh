set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_direct_api5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
OP=$1
KEY=$2
test ! -e "$R/$KEY.checker.json"
test ! -e "$R/$KEY.exit"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source "$R/../attention_generated_recovery5/native_install/vendors/attention_native_build5_transformer/bin/set_env.bash"
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$R/../attention_generated_recovery5/python:$O/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$R/torch_extensions"
export XDG_CACHE_HOME="$R/cache" ASCEND_CACHE_PATH="$R/ascend_cache" ASCEND_WORK_PATH="$R/ascend_work"
export ASCEND_PROCESS_LOG_PATH="$R/runtime_logs/$KEY" ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=0
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$R/tmp"
cd "$R"
export TMPDIR=/proc/self/cwd/tmp
set +e
npu-guard run --owner 'baseline700-20261026/attention_direct_api5' --wait 600 --timeout 300 -- /opt/tileops-ci/runtime/bin/python3 "$R/case_runner.py" --op "$OP" --case-id "$KEY" --library ops-transformer --output "$R/$KEY.checker.json" > "$R/$KEY.log" 2>&1
rc=$?
cat /run/lock/ascend-npu/device-0.lock > "$R/$KEY.guard.json"
printf '%s\n' "$rc" > "$R/$KEY.exit"
tail -45 "$R/$KEY.log"
exit "$rc"
