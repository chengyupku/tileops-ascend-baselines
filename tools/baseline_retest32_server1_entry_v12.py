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
if task['key'][0]=='tilelang-ascend':
    sys.path.insert(0,str(root/'.cache/all-pairs/remote_tla_compiler/python'))
    sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
    import setuptools
    from benchmarks.ascend_all_pairs.tla.native import runtime
    tl=runtime()
    cache=out/'jit_kernel_cache';cache.mkdir(exist_ok=False)
    tl.set_cache_dir(str(cache))
    assert Path(tl.get_cache_dir()).resolve()==cache.resolve()
if task['id']=='54ea178d41092467b54e':
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,repeat=True)
sys.argv=args
runpy.run_path(args[0],run_name="__main__")
