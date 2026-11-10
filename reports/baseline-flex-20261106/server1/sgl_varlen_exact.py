from dataclasses import replace
from pathlib import Path
import hashlib

def build(case):
    from benchmarks.ascend_all_pairs.sgl.attention import varlen
    from benchmarks.api import Implementation
    native = varlen(case)
    def run(q,k,v,cuq,cuk,*extra):
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        if len(qb)!=3 or len(kb)!=3 or qb[0]!=0 or kb[0]!=0 or qb[-1]!=q.shape[0] or kb[-1]!=k.shape[0]:
            raise ValueError('Exact two-request bounds required')
        if any(not (0 <= qb[j+1]-qb[j] <= kb[j+1]-kb[j]) for j in range(2)):
            raise ValueError('Unsupported query/KV lengths; no launch')
        return native.implementation.run(q,k,v,cuq,cuk,*extra)
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
