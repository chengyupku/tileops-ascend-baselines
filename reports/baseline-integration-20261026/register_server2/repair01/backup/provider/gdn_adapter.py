"""Exact-key GDN adapter delegating to source-pinned original ACLNN binding."""
from pathlib import Path
import json,os,sys
HERE=Path(__file__).resolve().parent;PRIVATE=Path("/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/transformer_gdn");CHUNK=PRIVATE/"chunk_package";KEYS=set(json.loads((HERE/"verified_keys.json").read_text())["GDNFwdOp"]);RECIPE=json.loads((CHUNK/"RUNTIME_RECIPE.json").read_text())
def _fallback(case):
 from benchmarks.ops.bench_ascend_classics import PROVIDERS as classic
 return classic["GDNFwdOp"]["ops-transformer"](case)
def gdn(case):
 if case.id not in KEYS:return _fallback(case)
 for k,v in RECIPE["required_environment"].items():os.environ[k]=v
 if str(CHUNK) not in sys.path:sys.path.insert(0,str(CHUNK))
 from binding import build
 return build(case)
