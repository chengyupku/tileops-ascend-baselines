"""Future parent launch only: isolated process, original checker, explicit backend.
Worker did not execute this module. No oracle, seed, tolerance or input override.
"""
import sys,os,json,runpy
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
import registration as M
# Guard and host identity before any device-capable imports.
M.environment();M.require_guard()
for p,h in M.metadata()['FROZEN.json']['sha256'].items():M.checked(p,h)
for p,h in M.metadata()['PINS.json']['runtime'].items():M.checked(p,h)
W=Path(M.ORIGINAL)
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
M.install(PROVIDERS) # this isolated evaluation process only; no registry file write
import torch
from backend_context import evaluation_backend
run=Path(os.environ['BMM_VARIANT_RUN_ROOT']);events=[]
def record(ev):
 events.append(ev)
 (run/'BACKEND_VARIANT.json').write_text(json.dumps(dict(events=events,admission='ORIGINAL_CPU_BMM_BACKEND_ADMISSION_20261027.json',default_backend_failure_retained=True),indent=2)+'\n')
checker=W/'TileOPs/scripts/ascend_pair_check.py'
sys.argv=[str(checker),'--op','BmmFwdOp','--case-id','square-4k-bfloat16','--library','ops-transformer','--output',str(run/'representative.json')]
with evaluation_backend(torch,record):
 runpy.run_path(str(checker),run_name='__main__')
