"""Select the original examples' supported AscendC public target."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry

def factory(case,library):
    def build():
        import torch
        from benchmarks.ascend_all_pairs.tla import native
        from benchmarks.ascend_all_pairs.tla.plans import ELEMENTWISE
        from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
        import dataclasses
        op=case._call.signature.name
        if library!='tilelang-ascend' or op not in ('CosFwdOp','IsnanFwdOp'):
            raise UnsupportedCase('Only original Cos/compare AscendC target routes')
        tl=native.runtime()
        output=Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else OUT
        cache=output/'original_ascendc_jit'
        cache.mkdir(exist_ok=True)
        tl.set_cache_dir(str(cache))
        x=case.inputs[0]
        size=x.numel(); rows=((size+4095)//4096)*64
        name='cos' if op=='CosFwdOp' else 'compare'
        module=native.load(ELEMENTWISE,name,'pass_configs')
        args=(rows,64,64,64) if op=='CosFwdOp' else (rows,64,64,64,'NE')
        kwargs={'dtype':'float32'}
        if op=='IsnanFwdOp':kwargs['out_dtype']='uint8'
        kernel=tl.compile(getattr(module,name)(*args,**kwargs),target='ascendc',out_idx=[-1],pass_configs=module.pass_configs)
        def run(x):
            packed=torch.nn.functional.pad(x.contiguous().float().reshape(-1),(0,rows*64-size)).reshape(rows,64)
            result=kernel(packed) if op=='CosFwdOp' else kernel(packed,packed)
            if op=='IsnanFwdOp':
                shifts=torch.arange(8,dtype=torch.uint8,device=x.device)
                result=((result.reshape(-1,1)>>shifts)&1).reshape(-1)
            return result.reshape(-1)[:size].reshape(x.shape).to(torch.bool if op=='IsnanFwdOp' else x.dtype)
        candidate=native.candidate(run,ELEMENTWISE+':'+name,
            'Original documented AscendC target, original builder/pass configuration and tile geometry. Exact widening, padding and output mask unpack only; no compiler/kernel edits.',ELEMENTWISE)
        return dataclasses.replace(candidate,source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))})
    return build

OUT=Path(sys.argv[sys.argv.index('--out')+1])
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
