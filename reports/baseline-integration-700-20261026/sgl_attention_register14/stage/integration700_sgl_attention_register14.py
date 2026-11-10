"""Private frozen SGL attention providers for integration700 exact 14-key scope."""
from attention_adapter import mla_decode, mha_paged
from attention_page16 import varlen
from dsa_adapter import build
from integration700_sgl_attention32 import write
from prefill_adapter import prefill_append
PROVIDERS = {
    "GQAPrefillPagedWithKVCacheFwdOp": prefill_append,
    "MHADecodePagedWithKVCacheFwdOp": mha_paged,
    "MLAKVCacheWriteFwdOp": write,
    "DSADecodeWithKVCacheFwdOp": build,
    "GQAVarlenFwdOp": varlen,
    "MLADecodeWithKVCacheFwdOp": mla_decode,
}
