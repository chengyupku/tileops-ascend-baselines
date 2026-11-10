"""Prove bounded tails and unchanged C for already measured full-tile configs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
subprocess.run([sys.executable,str(HERE/'tail_capacity_probe_v2.py')],check=True)
sys.path[:0]=[str(ROOT),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
import torch_npu
from candidate import _logroots_v7 as K
torch.npu.Event=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
OLD='bd4b4330c5f9ac1e377f6b9d7cbe0620c23201928feab59c3e75ddc6eb040766'
proof={'old_helper_sha256':OLD,'new_helper_sha256':hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest(),
       'guard_label':'tail-capacity-fix-and-compat','scope':'configuration comparisons only; final public acceptance still requires current helper hash',
       'comparisons':{}}
names={'LogFwdOp':'log','Log1pFwdOp':'log1p','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}
for directory in sorted((HERE/'search').glob('config15-*')):
    if not (directory/'result.json').exists():continue
    result=json.loads((directory/'result.json').read_text())
    hashes=[v for k,v in result['source_sha256'].items() if k.endswith('/_logroots.py')]
    if hashes!=[OLD]:continue
    op,size,dtype=directory.name.split('-')[-3:]
    n=int(size.removesuffix('M'))*1024*1024
    records=[]
    for trial in result['trials']:
        if trial['status']!='ok':continue
        config=trial['config'];assert n%config['tile']==0
        key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
        filename='candidate_'+key+'.c';old=(directory/filename).read_text()
        new=K.build(n,dtype,names[op],**config).get_kernel_source()
        assert new==old,(directory.name,config,'generated C differs')
        records.append({'config':config,'file':filename,'sha256':hashlib.sha256(new.encode()).hexdigest(),'exact_match':True})
    proof['comparisons'][directory.name]=records
torch.npu.synchronize()
(HERE/'search/source-equivalence-tailcap.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({'tail_checks':'passed','exact_C_comparisons':sum(map(len,proof['comparisons'].values())),
                  'old':OLD,'new':proof['new_helper_sha256']}),flush=True)
