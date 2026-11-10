"""Original tilelang FP8 API; skip only identity padding, keep original schedule."""
import sys, pathlib, torch, os
import torch.nn.functional as F

def factory(shape,cache):
    import benchmarks
    root=pathlib.Path('/opt/tileops-all-pairs/workspace')
    bp=str(root/'TileOPs/benchmarks')
    if bp not in benchmarks.__path__:benchmarks.__path__.append(bp)
    sys.path.insert(0,str(root/'ascend_baselines/tilelang'))
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    from benchmarks.ascend_baselines import local_function,source_hashes
    os.environ['TILELANG_CACHE_DIR']=str(cache)
    tl=_runtime()
    host='examples/ascend/example_per_token_cast_to_fp8.py'
    builder=local_function('tilelang','examples/ascend','example_per_token_cast_to_fp8','per_token_cast_to_fp8',host)
    hidden=shape[-1];rows=1
    for x in shape[:-1]:rows*=x
    assert hidden in (64,128)
    padded_rows=rows+(-rows)%32
    kernel=builder(padded_rows,128,mode='simt')
    def run(x):
        packed=x.float().reshape(rows,hidden)
        if hidden!=128 or padded_rows!=rows:packed=F.pad(packed,(0,128-hidden,0,padded_rows-rows))
        quant,scale=kernel(packed)
        return scale[:rows].reshape(x.shape[:-1]),quant[:rows,:hidden].contiguous().reshape(x.shape)
    sources=source_hashes(root/'ascend_baselines/tilelang'/host)|_runtime_hashes(tl)
    return run,sources
