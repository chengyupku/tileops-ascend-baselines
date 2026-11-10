"""Isolated ops-transformer providers; importing this module is device-free."""
from .adapters import bmm

PROVIDERS = {'BmmFwdOp': bmm}

# Server2 exact-key integrations. Imports are device-free; factories initialize
# original runtimes only when an accepted case is actually executed.
from .attention_adapter import gqa as _server2_gqa
from .gdn_adapter import gdn as _server2_gdn
PROVIDERS['GQADenseFwdOp'] = _server2_gqa
PROVIDERS['GDNFwdOp'] = _server2_gdn

# integration700_transformer_linear exact GDN keys
from .integration700_transformer_linear_gdn import wrap as _integration700_transformer_linear_wrap
PROVIDERS["GDNFwdOp"] = _integration700_transformer_linear_wrap(PROVIDERS.get("GDNFwdOp"))
