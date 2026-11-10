source /opt/tileops-all-pairs/control/baseline-env.sh
export NN_WORKSPACE=/opt/tileops-all-pairs/workspace
export PYTHONDONTWRITEBYTECODE=1
NN_PRIVATE_VENDOR=/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/vendor/allpairs_nn
export LD_LIBRARY_PATH="$NN_PRIVATE_VENDOR/op_api/lib:$NN_PRIVATE_VENDOR/op_proto/es/lib/linux/aarch64:$NN_PRIVATE_VENDOR/op_proto/lib/linux/aarch64:$NN_PRIVATE_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
exec "$@"
