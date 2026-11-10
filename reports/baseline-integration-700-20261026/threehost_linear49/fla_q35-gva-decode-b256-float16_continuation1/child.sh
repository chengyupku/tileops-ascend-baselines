set -e
source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh
export FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary"
export ASCEND_CLASSIC_FLA_PACKAGE="/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary/fla_npu"
export ASCEND_CUSTOM_OPP_PATH="/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary/fla_npu/opp/vendors/fla_npu_transformer:/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary/fla_npu/opp"
export LD_LIBRARY_PATH="/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/flash-linear-attention-npu/python_binary/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
exec /root/dyq/workspace/.venv/bin/python -B "/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_linear49/local_fla_entry.py" "q35-gva-decode-b256-float16" "/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_linear49/fla_q35-gva-decode-b256-float16_continuation1"
