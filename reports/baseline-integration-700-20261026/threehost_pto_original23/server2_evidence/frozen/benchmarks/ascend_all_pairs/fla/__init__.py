"""Original flash-linear-attention-npu providers; importing this is CPU-only."""


def cumsum(case):
    from .providers import cumsum as build
    return build(case)


PROVIDERS = {"CumsumFwdOp": cumsum}
