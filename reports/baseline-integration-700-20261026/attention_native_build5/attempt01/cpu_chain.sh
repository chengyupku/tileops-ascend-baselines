S=/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer
D=/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/dependencies/cann-cmake-master-058
cat "$D/AGENTS.md"
git -C "$D" rev-parse master-058; git -C "$D" remote -v
find "$S" -name AGENTS.md
sed -n '1,150p' "$S/cmake/scripts/util/ascendc_impl_build.py"
grep -nE 'python|cmake|build_host|build_kernel|torch_extension|CANN_3RD|check.*depend' "$S/build.sh"
find /usr/local/Ascend/cann-9.1.0 -path '*/tbe/__init__.py' -o -path '*/tbe/common/register/__init__.py'
find /opt/tileops-all-pairs/workspace/.cache /root/.cache -type f \( -name '*absl*.whl' -o -name 'numpy-1*.whl' -o -name 'build-*.whl' -o -name 'wheel-*.whl' -o -name 'cmake-master-058.tar.gz' \) 2>/dev/null | head -40
