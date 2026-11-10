S=/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer
D=/opt/tileops-all-pairs/workspace/.cache/all-pairs/resume22_server2_transformer_attention01/native_build/dependencies
ls -la "$D"
git -C "$D/cann-cmake-master-058" rev-parse HEAD; git -C "$D/cann-cmake-master-058" status --porcelain
find "$D" -maxdepth 2 -type f | head -35
sed -n '700,960p' "$S/build.sh"
grep -R -nE 'import torch|from torch|import acl|import te|import tbe' "$S/scripts" "$S/cmake" 2>/dev/null | head -60
cat "$S/cmake/third_party/opbase.cmake"
/opt/tileops-ci/runtime/bin/python3 - <<'PY'
import importlib.metadata as m
for n in ['torch','torch_npu','build','wheel','setuptools','packaging','pyyaml','absl-py','jinja2','numpy','decorator','sympy','scipy','attrs','protobuf','psutil']:
 try: print(n,m.version(n))
 except m.PackageNotFoundError: print(n,'MISSING')
PY
cmake --version | head -1
g++ --version | head -1
