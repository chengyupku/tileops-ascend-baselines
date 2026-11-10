B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt02
tail -15 "$B/native_build.log"
ps -eo pid,ppid,pgid,args | grep -E 'cc1plus|make -j|make.*prepare_build' | grep -v grep | tail -12
