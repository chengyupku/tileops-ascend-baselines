find /opt/tileops-all-pairs/workspace -maxdepth 5 -name ascend_pair_check.py
R=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt03/stage/source/build
find "$B/binary/ascend950/gen" -name '*.o' | wc -l
tail -4 "$R/compile1.log"
