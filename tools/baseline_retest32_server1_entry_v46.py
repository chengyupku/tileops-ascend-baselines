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
    os.environ['ASCEND_HOME_PATH']=str(root/'reports/baseline-retest32-20261030/server1/deployments/deepgemm_local_sdk')
    os.environ['ASCEND_TOOLKIT_HOME']=os.environ['ASCEND_HOME_PATH']
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
if task['key'][0]=='ops-transformer' and task['key'][1] in ('RMSNormFwdOp','RoPEFwdOp'):
    import baseline_retest32_server1_normrope_v46
elif task['key'][:2]==['ops-transformer','MoEPrePermuteFwdOp']:
    import baseline_retest32_server1_prepermute_v44
elif task['key'][:2]==['ops-transformer','MoEPostPermuteFwdOp']:
    import baseline_retest32_server1_postpermute_v43
elif task['key'][:2]==['ops-transformer','FusedTopKFwdOp']:
    import baseline_retest32_server1_fusedtopk_v42
elif task['key'][:2]==['ops-transformer','SoftmaxFwdOp']:
    import baseline_retest32_server1_softmax_v41
elif task['key'][:2]==['ops-transformer','MHCPostFwdOp']:
    import baseline_retest32_server1_transformer_mhc_v40
elif task['key'][:2]==['ops-transformer','AddFwdOp']:
    import baseline_retest32_server1_transformer_add_v36
elif task['key'][0]=='ops-transformer' and task['key'][1] in ('BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv3dFwdOp','Conv2dFwdOp'):
    import baseline_retest32_server1_transformer_graph_v33
elif task['key'][0]=='ops-transformer':
    import baseline_retest32_server1_transformer_v37
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
if task['key'][:2]==['flash-linear-attention-npu','GDNFwdOp'] and task['case']['params']['q'][1]=='float16':
    from benchmarks.ascend_all_pairs import registry
    from benchmarks.ascend_baselines import source_hashes
    import dataclasses
    original_factory=registry.candidate_factory
    def gdn_factory(case,library):
        make=original_factory(case,library)
        if library!='flash-linear-attention-npu' or case._call.signature.name!='GDNFwdOp':return make
        def build():
            candidate=make()
            original_run=candidate.implementation.run
            def run(q,k,v,g,beta,*rest,**kwargs):
                return original_run(q,k,v,g.float(),beta.float(),*rest,**kwargs)
            return dataclasses.replace(candidate,implementation=dataclasses.replace(candidate.implementation,run=run),source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))},adaptation=candidate.adaptation+'; per-call lossless FP16 gate/beta to FP32 for original Phase6 ABI')
        return build
    registry.candidate_factory=gdn_factory
sys.argv=args
runpy.run_path(args[0],run_name="__main__")
