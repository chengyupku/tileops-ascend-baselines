"""Apply only guarded, same-source, 15-window configuration comparisons."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
KERNELS=ROOT/'TileOPs/src/tileops/kernels/ascend/elementwise'
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
helper_hash=hashlib.sha256((KERNELS/'_logroots.py').read_bytes()).hexdigest()
def source_eligibility(directory,result):
    hashes=[v for k,v in result['source_sha256'].items() if k.endswith('/_logroots.py')]
    if hashes==[helper_hash]:return {'mode':'identical_helper_hash'}
    path=HERE/'search/source-equivalence-tailcap.json'
    if not path.exists():return None
    proof=json.loads(path.read_text())
    if proof['new_helper_sha256']!=helper_hash or hashes!=[proof['old_helper_sha256']]:return None
    receipts=[json.loads(p.read_text()) for p in (HERE/'requests').glob(proof['guard_label']+'_*/status.json')]
    if not receipts or max(receipts,key=lambda r:r['completed_at'])['guard_exit_code']!=0:return None
    records=proof['comparisons'].get(directory.name,[])
    trials=[r for r in result['trials'] if r['status']=='ok']
    if len(records)!=len(trials) or not records:return None
    for trial in trials:
        matches=[r for r in records if r['config']==trial['config'] and r['exact_match']]
        if len(matches)!=1:return None
        record=matches[0]
        if hashlib.sha256((directory/record['file']).read_bytes()).hexdigest()!=record['sha256']:return None
    return {'mode':'exact_generated_C_match_after_tail_capacity_fix','proof':str(path.relative_to(HERE)),
            'old_helper_sha256':proof['old_helper_sha256'],'current_helper_sha256':helper_hash}
selected=[];updates={}
for op,kind in [('LogFwdOp','log'),('Log1pFwdOp','log1p'),('SqrtFwdOp','sqrt'),('RsqrtFwdOp','rsqrt')]:
    configs={}
    for case in ('16M-float16','16M-bfloat16','16M-float32','256M-float16','256M-bfloat16'):
        choices=[]
        for directory in (HERE/'search').glob('config15*-'+op+'-'+case):
            if not (directory/'result.json').exists():continue
            result=json.loads((directory/'result.json').read_text())
            eligibility=source_eligibility(directory,result)
            if eligibility is None:continue
            requests=list((HERE/'requests').glob(directory.name+'_*/status.json'))
            if not requests:continue
            receipt=max((json.loads(path.read_text()) for path in requests),key=lambda row:row['completed_at'])
            if receipt['guard_exit_code']==0:choices.append((receipt['completed_at'],directory,result,receipt,eligibility))
        assert choices,op+' '+case+' has no current-source guarded comparison'
        _,directory,result,receipt,eligibility=max(choices,key=lambda item:item[0])
        label=directory.name
        trials=json.loads((directory/'trials.json').read_text())
        good=[r for r in trials if r['status']=='ok' and r['validation']=='passed']
        assert good,label+' has no qualified config'
        for r in good:
            profile=r['search_profile']
            assert profile['warmup']==3 and profile['repeats']==15 and not profile['event_used']
            assert len(profile['latency_samples_ms'])==15
            assert abs(sum(profile['latency_samples_ms'])/15-r['latency_ms'])<1e-12
        best=min(good,key=lambda r:r['latency_ms'])
        n,dtype=case.split('-');n=int(n.removesuffix('M'))*1024*1024
        value=tuple(best['config'][k] for k in ('tile','blocks','stages'))
        configs[(n,dtype)]=value
        if n==16*1024*1024:configs[dtype]=value
        selected.append({'op':op,'case_id':case,'config':best['config'],'comparison_us':best['latency_ms']*1000,
                         'evidence':str(directory.relative_to(HERE)),'guard_request':receipt['label'],
                         'source_eligibility':eligibility,
                         'qualification':'configuration selection only; final public entry M3/M4 required'})
    path=KERNELS/(kind+'.py');text=path.read_text()
    start=text.index('    configs = ');end=text.index('\n',start)
    lines=['    configs = {']+[f'        {key!r}: {value!r},' for key,value in configs.items()]+['    }']
    updates[path]=text[:start]+'\n'.join(lines)+text[end:]
print(json.dumps(selected,indent=2))
if a.apply:
    backup=HERE/'pre-config15-wrappers';backup.mkdir(exist_ok=True)
    for path,text in updates.items():
        assert not (backup/path.name).exists(),'Refuse to overwrite pre-selection source backup'
        shutil.copy2(path,backup/path.name);path.write_text(text)
    (HERE/'selected-configs-v3.json').write_text(json.dumps(selected,indent=2)+'\n')
    sources=list(updates)+[KERNELS/'_logroots.py']
    frozen={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'sources':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'reason':'Per-case 3/15 actual configuration comparisons; Log1p cubic small-input branch.'}
    (HERE/'frozen-v3.json').write_text(json.dumps(frozen,indent=2)+'\n')
    snapshot=HERE/'source-v3';snapshot.mkdir(exist_ok=True)
    for path in sources:shutil.copy2(path,snapshot/path.name)
