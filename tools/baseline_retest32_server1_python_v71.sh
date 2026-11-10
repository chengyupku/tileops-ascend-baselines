#!/bin/bash
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
lib="$1"
if [ "$lib" = ops-transformer ]; then
 export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_transformer/linear_recovery_transformer"
 case_op=$(python3 -c 'import json,sys;a=sys.argv[1:];print(json.load(open(a[a.index("--task")+1]))["key"][1])' "${@:2}")
 if [ "$case_op" = BmmFwdOp ] || [ "$case_op" = GroupedGemmFwdOp ] || [ "$case_op" = Conv1dFwdOp ] || [ "$case_op" = Conv3dFwdOp ] || [ "$case_op" = Conv2dFwdOp ] || [ "$case_op" = AddFwdOp ] || [ "$case_op" = MHCPostFwdOp ] || [ "$case_op" = MoEPostPermuteFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
 fi
 if [ "$case_op" = GQADenseFwdOp ] || [ "$case_op" = GQAVarlenFwdOp ] || [ "$case_op" = GQAPagedFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_gqa/vendors/gqapairs_transformer"
 fi
 if [ "$case_op" = SoftmaxFwdOp ]; then
  width=$(python3 -c 'import json,sys;a=sys.argv[1:];p=json.load(open(a[a.index("--task")+1]))["case"]["params"];print(p["x"][0][p["dim"]])' "${@:2}")
  export SERVER1_SOFTMAX_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/root"
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  if [ "$width" -gt 1024 ]; then
   export SERVER1_SOFTMAX_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/root"
   export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/vendors/selectionpairs_transformer"
  fi
 fi
 if [ "$case_op" = FusedTopKFwdOp ]; then
  selection=$(python3 -c 'import json,sys;a=sys.argv[1:];p=json.load(open(a[a.index("--task")+1]))["case"]["params"];print(int(not ((p["scoring_func"]=="softmax" and p["num_expert_group"]==1 and p["topk_group"]==1 and p["routed_scaling_factor"]==1.) or (p["scoring_func"]=="sigmoid" and not p["renormalize"]))))' "${@:2}")
  export SERVER1_FUSED_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/root"
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  if [ "$selection" = 1 ]; then
   export SERVER1_FUSED_GRAPH_ROOT="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/root"
   export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/vendors/selectionpairs_transformer"
  fi
 fi
 if [ "$case_op" = MoEPrePermuteFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_prepermute/vendors/prepermutepairs_transformer"
 fi
 if [ "$case_op" = FusedQKNormRoPEFwdOp ] || [ "$case_op" = InstanceNormFwdOp ] || [ "$case_op" = RMSNormFwdOp ] || [ "$case_op" = RoPEFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_normrope_v47/vendors/normropepairs_transformer"
 fi
 if [ "$case_op" = AmaxFwdOp ] || [ "$case_op" = ReluFwdOp ] || [ "$case_op" = ClampScalarFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_attention_v51/vendors/attentionpairs_transformer"
 fi
 if [ "$case_op" = RoPELlama31FwdOp ] || [ "$case_op" = YaRNFwdOp ] || [ "$case_op" = LongRoPEFwdOp ] || [ "$case_op" = RoPENeoxPositionIdsFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_normrope_v47/vendors/normropepairs_transformer"
 fi
 if [ "$case_op" = MLADecodeWithKVCacheFwdOp ] || [ "$case_op" = MHADecodePagedWithKVCacheFwdOp ] || [ "$case_op" = DSADecodeWithKVCacheFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_gqa/vendors/gqapairs_transformer"
 fi
 if [ "$case_op" = GemmW4A16FwdOp ] || [ "$case_op" = GemmFP8FwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
 fi
 export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"
 if [ "$case_op" = ArgminFwdOp ] || [ "$case_op" = TopKSelectFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$root/reports/baseline-retest32-20261030/server1/deployments/ops_selection/vendors/selectionpairs_transformer:$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer:$root/reports/baseline-retest32-20261030/server1/deployments/ops_attention_v51/vendors/attentionpairs_transformer"
 fi
 if [ "$case_op" = HadamardTransformFwdOp ] || [ "$case_op" = GemmFwdOp ] || [ "$case_op" = CumsumFwdOp ] || [ "$case_op" = SSDChunkCouplingFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_graph/vendors/allpairs_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR:$root/reports/baseline-retest32-20261030/server1/deployments/ops_attention_v51/vendors/attentionpairs_transformer"
 fi
 if [ "$case_op" = PagedKVCacheWriteFwdOp ] || [ "$case_op" = MLAKVCacheWriteFwdOp ] || [ "$case_op" = PagedKVCacheGatherFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_cache/vendors/cachepairs_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR:$root/reports/baseline-retest32-20261030/server1/deployments/ops_normrope_v47/vendors/normropepairs_transformer"
 fi
 if [ "$case_op" = GQAPrefillPagedWithKVCacheFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_cache/vendors/cachepairs_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR:$root/reports/baseline-retest32-20261030/server1/deployments/ops_gqa/vendors/gqapairs_transformer"
 fi
 if [ "$case_op" = CausalConv1dFwdOp ] || [ "$case_op" = CausalConv1dUpdateFwdOp ]; then
  export OLD1076_TRANSFORMER_VENDOR="$root/reports/baseline-retest32-20261030/server1/deployments/ops_causal/vendors/causalpairs_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$OLD1076_TRANSFORMER_VENDOR"
 fi
 export ASCEND_CLASSIC_TRANSFORMER_VENDOR="$OLD1076_TRANSFORMER_VENDOR"
 export LD_LIBRARY_PATH="$OLD1076_TRANSFORMER_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
 exec /opt/tileops-all-pairs/bin/allpairs-python "$root/tools/baseline_retest32_server1_entry_v71.py" "${@:2}"
fi
if [ "$lib" = tilelang-ascend ]; then
 export INTEGRATION700_TLA_ROUTE=legacy
 export TILELANG_LIBRARY_PATH="$root/.cache/all-pairs/remote_tla_compiler/handoff/lib"
 export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
 export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
 exec /opt/tileops-all-pairs/bin/allpairs-python "$root/tools/baseline_retest32_server1_entry_v71.py" "${@:2}"
fi
if [ "$lib" = vllm-ascend ]; then
 exec /opt/tileops-all-pairs/bin/allpairs-python "$root/tools/baseline_retest32_server1_entry_v71.py" "${@:2}"
fi
if [ "$lib" = flash-linear-attention-npu ]; then
 export ASCEND_CLASSIC_FLA_PACKAGE="$root/.cache/old1076/fla/fla_npu"
 case_op=$(python3 -c 'import json,sys; a=sys.argv[1:]; print(json.load(open(a[a.index("--task")+1]))["key"][1])' "${@:2}")
 recurrent=$(python3 -c 'import json,sys; a=sys.argv[1:]; t=json.load(open(a[a.index("--task")+1])); op=t["key"][1]; q=t["case"]["params"].get("q",[[0,0]])[0]; print(int(op in ("DeltaNetRecurrentFwdOp","KDAFwdOp") or (op in ("GDNFwdOp","DeltaNetFwdOp") and t["case"]["params"]["q"][1]=="bfloat16" and len(q)==4 and (q[1]<=8 or (op=="GDNFwdOp" and t["case"]["params"].get("use_gate_in_kernel"))))))' "${@:2}")
 if [ "$recurrent" = 1 ]; then
  export ASCEND_CLASSIC_FLA_PACKAGE="$root/reports/baseline-retest32-20261030/server1/deployments/fla_recurrent/fla_npu"
 fi
 if [ "$case_op" = CausalConv1dFwdOp ] || [ "$case_op" = CausalConv1dUpdateFwdOp ]; then
  export ASCEND_CLASSIC_FLA_PACKAGE="$root/reports/baseline-retest32-20261030/server1/deployments/fla_causal/fla_npu"
 fi
 if [ "$case_op" = DeltaNetChunkBwdOp ]; then
  export ASCEND_CLASSIC_FLA_PACKAGE="$root/reports/baseline-retest32-20261030/server1/deployments/fla_backward/fla_npu"
 fi
 if [ "$case_op" = SSDChunkCumsumFwdOp ]; then
  export ASCEND_CLASSIC_FLA_PACKAGE=$root/reports/baseline-retest32-20261030/server1/deployments/fla_gate/fla_npu
 fi
 export ASCEND_CUSTOM_OPP_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer:$ASCEND_CLASSIC_FLA_PACKAGE/opp"
 export LD_LIBRARY_PATH="$ASCEND_CLASSIC_FLA_PACKAGE/opp/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
exec bash "$root/tools/baseline_cache_python.sh" "$lib" "$root/tools/baseline_retest32_server1_entry_v71.py" "${@:2}"
