set -e
O=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5
cd "$O/attempt03/stage/source/build"
for target in optiling_compat gen_version_info modify_vendor; do
 /usr/bin/make -j1 -f "CMakeFiles/$target.dir/build.make" "$target"
done
