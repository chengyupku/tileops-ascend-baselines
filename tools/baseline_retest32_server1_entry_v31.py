"""Private environment path bootstrap; invoked entirely within host guard."""
import hashlib,json,os,runpy,sys
from pathlib import Path
package=os.environ.get("ASCEND_CLASSIC_FLA_PACKAGE")
if package:
    sys.path.insert(0,str(Path(package).resolve().parent))
args=sys.argv[1:]
out=Path(args[args.index("--out")+1])
entry=Path(__file__).resolve()
receipt=out/"launch_environment.json"
receipt.write_text(json.dumps({"launcher":str(entry),"launcher_sha256":hashlib.sha256(entry.read_bytes()).hexdigest(),"package":package,"package_parent_in_sys_path":str(Path(package).resolve().parent) in sys.path if package else None}))
os.environ["BASELINE_CACHE_LAUNCH_RECEIPT"]=str(receipt)
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'TileOPs'),str(root/'TileOPs/src'),str(root)]
task=json.loads(Path(args[args.index('--task')+1]).read_text())
if task['key'][0]=='DeepGEMM-Ascend':
    import importlib.util
    selected=Path('/opt/tileops-ci/runtime/lib/python3.12/site-packages/tilelang')
    sys.path.append(str(selected.parent))
    assert (selected/'ascend/__init__.py').is_file()
    assert 'tilelang' not in sys.modules
    spec=importlib.util.spec_from_file_location('tilelang',selected/'__init__.py',submodule_search_locations=[str(selected)])
    module=importlib.util.module_from_spec(spec)
    sys.modules['tilelang']=module
    spec.loader.exec_module(module)
if task['key'][:2]==['ops-transformer','GroupedGemmFwdOp'] and task['case']['params']['a'][1]=='bfloat16':
    import torch
    torch.backends.mkldnn.enabled=False
    env_record=json.loads(receipt.read_text())
    env_record['cpu_mkldnn_enabled']=torch.backends.mkldnn.enabled
    env_record['cpu_reference_environment_reason']='ARM SVE256 oneDNN BF16 mm diverged from FP64; unchanged torch.mm fallback and native candidate matched FP64 within original tolerance.'
    receipt.write_text(json.dumps(env_record))

if task['key'][:2]==['flash-linear-attention-npu','SSDChunkCumsumFwdOp']:
    import benchmarks.ascend_all_pairs.fla as fla
    # Restore only the exact existing original provider; no global readiness marker.
    fla.PROVIDERS['SSDChunkCumsumFwdOp']=fla._include_host_work(fla.chunk_cumsum)
if task['key'][:2]==['flash-linear-attention-npu','DeltaNetChunkFwdOp']:
    import benchmarks.ascend_all_pairs.fla as fla
    fla.PROVIDERS['DeltaNetChunkFwdOp']=fla._include_host_work(fla.delta_chunk_forward)
if task['key'][:2]==['flash-linear-attention-npu','DeltaNetChunkBwdOp']:
    import benchmarks.ascend_all_pairs.fla as fla
    fla.PROVIDERS['DeltaNetChunkBwdOp']=fla._include_host_work(fla.delta_backward)
if task['key'][0]=='ops-transformer' and task['key'][1] in ('BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv3dFwdOp'):
    import baseline_retest32_server1_transformer_graph_v30
elif task['key'][0]=='ops-transformer':
    import baseline_retest32_server1_transformer_v15
if task['key'][0]=='tilelang-ascend':
    sys.path.insert(0,str(root/'.cache/all-pairs/remote_tla_compiler/python'))
    sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
    import setuptools
    from benchmarks.ascend_all_pairs.tla.native import runtime
    tl=runtime()
    cache=out/'jit_kernel_cache';cache.mkdir(exist_ok=False)
    tl.set_cache_dir(str(cache))
    assert Path(tl.get_cache_dir()).resolve()==cache.resolve()
if task['key'][1] in ('GLAFwdOp','KDAFwdOp'):
    import faulthandler
    faulthandler.enable()
    # Periodic watchdog removed after three profiler SIGSEGVs; fatal handler retained.
sys.argv=args
runpy.run_path(args[0],run_name="__main__")
