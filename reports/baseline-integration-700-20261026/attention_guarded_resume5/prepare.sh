set -e
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_guarded_resume5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03
G="$B/stage/source/build/binary/ascend950/gen"
cat "$G/compile_pids.txt"
while read -r pid; do test ! -e "/proc/$pid" || { echo "old PID exists $pid"; exit 1; }; done < "$G/compile_pids.txt"
cp "$G/compile_pids.txt" "$R/compile_pids.before.txt"
: > "$G/compile_pids.txt"
test ! -e "$G/compile_stop.flag"
rg -n 'MAX.*(JOB|THREAD)|PARALLEL|TILINGKEY_PAR_COMPILE' /usr/local/Ascend/cann-9.1.0/python/site-packages/ /usr/local/Ascend/cann-9.1.0/compiler/python/site-packages/ --glob '*opc*' --glob '*.py' | head -60
