"""Validate a candidate through the existing public operator and unchanged checkers."""
import sys
import runpy
from pathlib import Path
import torch
import torch_npu
import broadcast_candidate as bool_candidate
from tileops.kernels.ascend.elementwise import bitwise_binary
from tileops.backend import register_kernel_type
assert torch.npu.get_device_limit(0)=={'cube_core_num':32,'vector_core_num':64}
bool_candidate.AscendBitwiseAndKernel.preferred_over=frozenset({'ascend'})
register_kernel_type('BitwiseAndFwdOp','ascend_bool_candidate',bool_candidate.AscendBitwiseAndKernel)
script,*args=sys.argv[1:]
sys.argv=[script,*args]
runpy.run_path(script,run_name='__main__')
