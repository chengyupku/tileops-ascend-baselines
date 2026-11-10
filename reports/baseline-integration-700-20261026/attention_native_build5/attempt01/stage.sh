set -e
B=/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_native_build5/attempt01
S=/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer
D=/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/dependencies/cann-cmake-master-058
mkdir -p "$B"
test ! -e "$B/stage"
mkdir -p "$B/stage/source" "$B/stage/python" "$B/stage/wheels" "$B/stage/tmp" "$B/stage/bin" "$B/stage/install"
tar -C "$S" --exclude='./build' --exclude='./build_out' --exclude='./.git' -cf "$B/original_source.tar" .
sha256sum "$B/original_source.tar"
tar -C "$B/stage/source" -xf "$B/original_source.tar"
git -C "$D" rev-parse 'master-058^{}'
git -C "$D" archive 'master-058^{}' > "$B/cann-cmake-master-058.tar"
sha256sum "$B/cann-cmake-master-058.tar"
mkdir -p "$B/stage/source/third_party/cann-cmake"
tar -C "$B/stage/source/third_party/cann-cmake" -xf "$B/cann-cmake-master-058.tar"
cp /opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/build_deps/wheels/wheel-0.45.1-py3-none-any.whl "$B/stage/wheels/"
ls /opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/build_deps/wheels
sed -n '2430,2545p' "$S/build.sh"
sed -n '1070,1135p' "$S/build.sh"
cat /usr/local/Ascend/cann-9.1.0/python/site-packages/tbe/__init__.py
find /usr/local/Ascend/cann-9.1.0/python/site-packages/asc_op_compile_base -maxdepth 2 -name '__init__.py' | head
