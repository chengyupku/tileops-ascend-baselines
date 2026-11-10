S=/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer
cd "$S"
cat requirements.txt cmake/fetch_cann_cmake.cmake version.cmake
sed -n '1,180p' CMakeLists.txt
rg -n 'python|pip|cmake|source |build_module' build.sh
sed -n '1,145p' torch_extension/setup.py
cat torch_extension/pyproject.toml
cat /proc/self/cgroup
cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
find /opt/tileops-all-pairs/workspace/.cache /root/.cache -iname '*cann*cmake*' -o -name 'build-*.whl' -o -name 'pybind11-*.whl' 2>/dev/null | head -60
