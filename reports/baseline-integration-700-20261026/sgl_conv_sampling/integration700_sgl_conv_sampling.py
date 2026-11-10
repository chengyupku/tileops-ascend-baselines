"""Private source-faithful route descriptions for SGL Conv2d/chain sampling.
Numerical dispatch is intentionally disabled until the host checker runtime is available.
"""
from pathlib import Path
ROOT=Path("/root/dyq/workspace/ascend_baselines/sgl-kernel-npu")
ROUTES={
 "Conv2dFwdOp": {"api":"torch.ops.npu.sgemmc_expand", "source":"csrc/lora/op_host/sgemmc_expand.cpp", "binding":"NCHW 1x1 -> [N*H*W,Cin], weight [1,Cout,Cin], y [M,Cout]"},
 "ChainSpeculativeSamplingFwdOp": {"api":"sgl_kernel_npu.sample.chain_speculative_sampling.chain_speculative_sampling_triton", "source":"python/sgl_kernel_npu/sgl_kernel_npu/sample/chain_speculative_sampling.py", "binding":"full predicts/accept_index/accept_token_num plus candidate/retrieval/uniform/probability metadata"},
}
PROVIDERS={}
