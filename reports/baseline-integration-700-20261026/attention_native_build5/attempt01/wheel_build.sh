set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
B="$R/attempt02"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$R/attempt01/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages"
export PATH="$B/stage/bin:$PATH" TMPDIR="$B/stage/tmp"
cd "$B/stage/source"
timeout 120 bash build.sh --torch_extension --soc=ascend950 > "$B/wheel_build.log" 2>&1
find build_out -maxdepth 1 -name '*.whl' -exec sha256sum {} \;
tail -5 "$B/wheel_build.log"
