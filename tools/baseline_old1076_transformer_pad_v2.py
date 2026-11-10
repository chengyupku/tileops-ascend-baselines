"""Top-level original ChunkKdaFwd API adaptation on the existing ARM deployment."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'.cache/old1076/linear_recovery_transformer'
if os.environ.get('OLD1076_TRANSFORMER_VENDOR')!=str(VENDOR):
    env=dict(os.environ,OLD1076_TRANSFORMER_VENDOR=str(VENDOR),ASCEND_CUSTOM_OPP_PATH=str(VENDOR))
    env['LD_LIBRARY_PATH']=str(VENDOR/'op_api/lib')+':'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
sys.path[:0]=[str(ROOT/'tools/old1076_transformer'),str(ROOT/'tools'),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
import dataclasses,runpy
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes
from baseline_old1076_transformer_pad_dispatch_v2 import dispatch

def factory(case,library):
    if library!='ops-transformer' or case._call.signature.name not in ('GDNFwdOp','KDAFwdOp','DeltaNetFwdOp','DeltaNetRecurrentFwdOp'):raise ValueError('Outside owned ChunkKda interface scope')
    def build():
        c=dispatch(case)
        return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__),ROOT/'tools/baseline_old1076_transformer_pad_dispatch_v2.py')})
    return build
if __name__=='__main__':
    registry.candidate_factory=factory
    runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
