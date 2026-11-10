# Source this file in the dedicated remote runner before any guarded command.
source /usr/local/Ascend/cann/set_env.sh
source /opt/tileops-ci/runtime/bin/activate
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
export USE_ASCEND=ON USE_CUDA=OFF USE_ROCM=OFF
export ASCEND_NPU_ARCH=dav-3510
export TILELANG_CACHE_DIR=/opt/tileops-ci/cache/tilelang
