set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
A="$R/attempt01"; B="$R/attempt03"
mkdir -p "$B/stage/source" "$B/stage/bin" "$B/stage/tmp"
tar -C "$B/stage/source" -xf "$A/original_source.tar"
cp -a "$A/stage/source/third_party/cann-cmake" "$A/stage/source/third_party/ops-tensor" "$B/stage/source/third_party/"
cat > "$B/stage/bin/make" <<'PY'
#!/opt/tileops-ci/runtime/bin/python3
import os,sys
args=[]; skip=False
for a in sys.argv[1:]:
 if skip: skip=False; continue
 if a in ('-j','--jobs'): skip=True; continue
 if a.startswith('-j') or a.startswith('--jobs='): continue
 args.append(a)
os.execv('/usr/bin/make',['/usr/bin/make',*args,'-j1'])
PY
chmod +x "$B/stage/bin/make"
ln -s make "$B/stage/bin/gmake"
ln -s /opt/tileops-ci/runtime/bin/python3 "$B/stage/bin/python3"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$A/stage/python:/opt/tileops-ci/runtime/lib/python3.12/site-packages:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$B/stage/torch_extensions" MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$B/stage/tmp" XDG_CACHE_HOME="$B/stage/cache" CCACHE_DIR="$B/stage/ccache" ASCEND_CACHE_PATH="$B/stage/ascend_cache" ASCEND_WORK_PATH="$B/stage/ascend_work"
export PATH="$B/stage/bin:$PATH"
cd "$B/stage/source"
set +e
t0=$(date +%s)
timeout --signal=TERM --kill-after=10 1650 bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --vendor_name=attention_native_build5 -j1 > "$B/native_build.log" 2>&1
rc=$?; t1=$(date +%s)
printf '{"exit_code":%s,"elapsed_seconds":%s}\n' "$rc" "$((t1-t0))" > "$B/native_build_receipt.json"
tail -65 "$B/native_build.log"
cat "$B/native_build_receipt.json"
exit "$rc"
