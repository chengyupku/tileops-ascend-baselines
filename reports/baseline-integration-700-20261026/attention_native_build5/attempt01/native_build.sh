set -e
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01
STAGE="$B/stage"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$STAGE/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$STAGE/torch_extensions" MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 OMP_NUM_THREADS=8
export TMPDIR="$STAGE/tmp" XDG_CACHE_HOME="$STAGE/cache" CCACHE_DIR="$STAGE/ccache" ASCEND_CACHE_PATH="$STAGE/ascend_cache" ASCEND_WORK_PATH="$STAGE/ascend_work"
ln -s /opt/tileops-ci/runtime/bin/python3 "$STAGE/bin/python3"
export PATH="$STAGE/bin:$PATH"
cd "$STAGE/source"
python3 - <<'PY' > "$B/requirements_check.json"
import importlib.metadata as m,json
from packaging.requirements import Requirement
rows=[]
for line in open('requirements.txt'):
 r=Requirement(line.strip()); v=m.version(r.name); rows.append(dict(name=r.name,version=v,required=str(r.specifier),ok=v in r.specifier))
print(json.dumps(rows,indent=2)); assert all(x['ok'] for x in rows)
PY
set +e
t0=$(date +%s)
timeout --signal=TERM --kill-after=10 1800 bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --vendor_name=attention_native_build5 -j8 > "$B/native_build.log" 2>&1
rc=$?
t1=$(date +%s)
printf '{"exit_code":%s,"elapsed_seconds":%s}\n' "$rc" "$((t1-t0))" > "$B/native_build_receipt.json"
tail -100 "$B/native_build.log"
cat "$B/native_build_receipt.json"
exit "$rc"
