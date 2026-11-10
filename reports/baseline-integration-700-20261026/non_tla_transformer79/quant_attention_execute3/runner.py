import sys,os,json,hashlib,runpy
from pathlib import Path
P=Path(__file__).resolve().parent
W=Path('/opt/tileops-all-pairs/workspace')
SDK=W/'.cache/all-pairs/resume22_server2_tilekernels_adaptive_norm02/sdk_view_v10'
sys.dont_write_bytecode=True
sys.path[:0]=[str(P),str(P/'wheel'),str(W/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'),str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(SDK/'python/site-packages'),str(SDK/'opp/built-in/op_impl/ai_core/tbe')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
expected=json.loads((P/'PRE_USE_IDENTITY.json').read_text())
for path,identity in expected.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==identity['sha256'],path
from benchmarks.ascend_all_pairs import ops_transformer
from registration import provider,event
ops_transformer.PROVIDERS['GQAVarlenFwdOp']=provider
cid=sys.argv[1];os.environ['CASE_ID']=cid
sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op','GQAVarlenFwdOp','--case-id',cid,'--library','ops-transformer','--output',str(P/(cid+'.checker.json'))]
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
finally:
    torch=sys.modules.get('torch')
    if torch is not None and hasattr(torch,'npu') and torch.npu.is_initialized():
        event('final_sync_enter')
        torch.npu.synchronize()
        event('final_sync_completed')
    (P/(cid+'.maps')).write_text(Path('/proc/self/maps').read_text())
    drift=[path for path,v in expected.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=v['sha256']]
    (P/(cid+'.identity_post.json')).write_text(json.dumps({'drift':drift}))
