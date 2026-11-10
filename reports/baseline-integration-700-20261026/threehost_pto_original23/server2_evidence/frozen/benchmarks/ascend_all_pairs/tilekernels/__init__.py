"""TileKernels providers. Importing this module never imports a device runtime."""
from .adapters import silu_and_mul

PROVIDERS = {"SiluAndMulFwdOp": silu_and_mul}
