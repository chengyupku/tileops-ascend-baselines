B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03
ps -eo pid,ppid,pgid,etimes,comm,args | grep -E 'bisheng|asc_opc|cc1plus' | grep -v grep | head -12
find "$B/stage/source/build/binary" -name '*.log' | head -10
