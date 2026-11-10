"""SGL's own implementations; importing this registry never imports torch/NPU."""
from .elementwise import sigmoid, silu

PROVIDERS = {'SigmoidFwdOp': sigmoid, 'SiluFwdOp': silu}
