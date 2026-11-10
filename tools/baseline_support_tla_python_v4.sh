#!/usr/bin/env bash
# Environment selection only. Entire invocation belongs inside npu-guard.
set -e
baseline_cache_library="$1"
shift
baseline_cache_root="$(cd "$(dirname "$0")/.." && pwd)"
# CPU-only deployment dispatch; preserve existing default routes unchanged.
if [ "$baseline_cache_root" = /root/dyq/workspace ] && [ "${BASELINE_CACHE_EXACT_ENV_SELECTED:-}" != 1 ]; then
    exec /usr/bin/python3 -I -S "$baseline_cache_root/tools/baseline_cache_exact_env_v4.py" "$baseline_cache_library" "$@"
fi
export PYTHONDONTWRITEBYTECODE=1
export FLA_NPU_DISABLE_PTH=1
if [ -x /opt/tileops-all-pairs/bin/allpairs-python ] && [ "$baseline_cache_root" = /opt/tileops-all-pairs/workspace ]; then
    if [ "$baseline_cache_library" = tilelang-ascend ]; then
        export INTEGRATION700_TLA_ROUTE=legacy
        export TILELANG_LIBRARY_PATH="$baseline_cache_root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
        export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
        export PATH="$baseline_cache_root/.cache/all-pairs/remote_tla_compiler/handoff/bin:$PATH"
        export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
        exec /opt/tileops-all-pairs/bin/allpairs-python "$baseline_cache_root/tools/baseline_support_tla_remote_launch_v4.py" "$baseline_cache_library" "$@"
    fi
    if [ "$baseline_cache_library" = vllm-ascend ]; then
        export VLLM_ALL_PAIRS_EXTENSION="$baseline_cache_root/.cache/all-pairs/vllm/install/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so"
        export VLLM_ALL_PAIRS_OPP="$baseline_cache_root/.cache/all-pairs/vllm/install/vendors/custom_transformer"
        export ASCEND_CUSTOM_OPP_PATH="$VLLM_ALL_PAIRS_OPP"
        export LD_LIBRARY_PATH="$VLLM_ALL_PAIRS_OPP/op_api/lib:${LD_LIBRARY_PATH:-}"
    fi
    if [ "$baseline_cache_library" = sgl-kernel-npu ]; then
        # Existing original ARM64 library was built with GCC13. Keep its C++
        # runtime private; the ascend-named path is bound by native cache hashes.
        export LD_LIBRARY_PATH="$baseline_cache_root/.cache/all-pairs/sgl-ascend-runtime:${LD_LIBRARY_PATH:-}"
    fi
    if [ "$baseline_cache_library" = ops-math ]; then
        source "$baseline_cache_root/.cache/all-pairs/remote_ops_math/remote_env.sh"
        export ASCEND_CUSTOM_OPP_PATH="$baseline_cache_root/.cache/all-pairs/remote_ops_math/graph-runtime/vendors/allpairs_math_math"
    fi
    exec /opt/tileops-all-pairs/bin/allpairs-python "$@"
fi
source "$baseline_cache_root/env.sh"
if [ "$baseline_cache_library" = flash-linear-attention-npu ]; then
    export FLA_NPU_DISABLE_PTH=0
fi
case "$baseline_cache_library" in
    ops-nn)
        # Exact vendor is selected by CPU metadata below, before venv startup.
        export ASCEND_CUSTOM_OPP_PATH=""
        ;;
    flash-linear-attention-npu)
        exec /usr/bin/python3 -I -S "$baseline_cache_root/tools/baseline_cache_env.py" --python "$baseline_cache_root/.venv/bin/python" -- "$@"
        ;;
    ops-math)
        export ASCEND_CUSTOM_OPP_PATH=""
        ;;
    ops-transformer)
        export ASCEND_CUSTOM_OPP_PATH=""
        for baseline_cache_vendor in "$baseline_cache_root/.cache/all-pairs/ops_transformer/install/vendors/"*; do
            export ASCEND_CUSTOM_OPP_PATH="$baseline_cache_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
        done
        export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CUSTOM_OPP_PATH:$baseline_cache_root/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer"
        ;;
    tilelang-ascend)
        # Do not revive the retired compiler launcher. Provider must resolve an
        # intact original deployment, otherwise this task is recorded unavailable.
        export INTEGRATION700_TLA_ROUTE=legacy
        exec "$baseline_cache_root/.venv/bin/python" "$baseline_cache_root/tools/baseline_support_tla_cache_launch_v4.py" "$@"
        ;;
    *) exec "$baseline_cache_root/.tools/baselines/baseline-run" "$baseline_cache_library" python "$@" ;;
esac
case "$baseline_cache_library" in
    ops-math|ops-nn|ops-transformer)
        export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
        export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH" ASCEND_AICPU_PATH="$ASCEND_HOME_PATH" ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
        ;;
esac
if [ "$baseline_cache_library" = ops-nn ] || [ "$baseline_cache_library" = ops-math ]; then
    exec /usr/bin/python3 -I -S "$baseline_cache_root/tools/baseline_cache_env.py" --python "$baseline_cache_root/.venv/bin/python" -- "$@"
fi
for baseline_cache_vendor in ${ASCEND_CUSTOM_OPP_PATH//:/ }; do
    export LD_LIBRARY_PATH="$baseline_cache_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
done
exec "$baseline_cache_root/.venv/bin/python" "$@"
