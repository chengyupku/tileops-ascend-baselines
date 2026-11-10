R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build
 tail -8 "$R/compile1.log"
ps -eo pid,ppid,etime,pcpu,args | grep -E 'asc_opc|bisheng|ccec|kernel_meta|cc1' | head -15
ls "$B/binary/ascend950/gen"/*.done
cat "$B/CMakeFiles/ops_transformer_kernel.dir/build.make" | tail -35
