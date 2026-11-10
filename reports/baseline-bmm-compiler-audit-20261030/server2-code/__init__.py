"""Lazy ops-nn providers. Importing this module never imports a device runtime."""
from .adapters import adaptive_max_pool

PROVIDERS = {
    "AdaptiveMaxPool2dFwdOp": adaptive_max_pool,
    "AdaptiveMaxPool2dIndicesFwdOp": adaptive_max_pool,
}
