O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
B=$O/attempt03/stage/source/build
cat "$B/CMakeFiles/ops_config_ascend950.dir/build.make" | tail -45
cat "$B/CMakeFiles/ops_transformer_config.dir/build.make" | tail -35
ls "$O/attempt01/stage/wheels"
sed -n '1,160p' "$O/attempt03/stage/source/experimental/torch_extension/cann_ops_transformer/__init__.py"
find "$O/attempt03/stage/source" -maxdepth 3 -iname '*torch*'
