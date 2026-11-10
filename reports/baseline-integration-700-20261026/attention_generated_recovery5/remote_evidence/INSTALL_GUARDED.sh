set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$R/python:$O/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PATH="$O/attempt03/stage/bin:$PATH"
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/cache" ASCEND_WORK_PATH="$R/ascend_work"
cd "$R/install_payload"
set +e
npu-guard run --owner 'baseline700-20261026/attention_generated_recovery5' --wait 600 --timeout 600 -- bash ./install.sh --install-path="$R/native_install" > "$R/native_install.log" 2>&1
rc=$?
cat /run/lock/ascend-npu/device-0.lock > "$R/native_install.guard.json"
printf '%s\n' "$rc" > "$R/native_install.exit"
tail -40 "$R/native_install.log"
exit "$rc"
