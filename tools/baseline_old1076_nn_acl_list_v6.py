"""Bind original subset API before use; supply stock CANN host dependencies."""
from pathlib import Path
import ctypes,os
ROOT=Path(__file__).resolve().parents[1]
# The subset API has undefined references to unrelated original CANN host ops.
# Prefer its own symbols with DEEPBIND; original _owned_native still verifies
# both called ACLNN exports against the subset ELF with nm and dladdr.
_cann_host=ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libopapi.so',mode=os.RTLD_NOW|os.RTLD_GLOBAL)
_subset=ctypes.CDLL(str(Path(os.environ['OPS_NN_VENDOR'])/'op_api/lib/libcust_opapi.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL|os.RTLD_DEEPBIND)
exec(compile((ROOT/'tools/baseline_old1076_nn_acl_list_v5.py').read_text(),str(Path(__file__)),'exec'))
