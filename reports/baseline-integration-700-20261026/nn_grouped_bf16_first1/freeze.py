from pathlib import Path
import json,hashlib,re,sys,datetime
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace');out=P/sys.argv[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((out/'checker.json').read_text());g=json.loads((out/'GUARD_RESULT.json').read_text())
assert c['status']=='verified' and g['returncode']==0
assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
assert c['independent_cpu_reference'] and c['reference_device']=='cpu'
assert c['input_seed']==int.from_bytes(hashlib.sha256(f"GroupedGemmFwdOp::{c['case_id']}".encode()).digest()[:4],'big')
assert sha(P/'adapter.py')=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
for k,v in c['source_sha256'].items():assert sha(W/k)==v,k
for k,v in c['oracle_source_sha256'].items():assert sha(P/'snapshot'/k)==v,k
compares=json.loads((W/'reports/baseline-integration-700-20261026/nn_recovery02/SOURCE_COMPARE.json').read_text())
for r in compares:assert r['identical'] and sha(Path(r['source']))==sha(Path(r['build']))==r['sha256']
events=[json.loads(l) for l in (out/'native.jsonl').read_text().splitlines()]
compiles=[e for e in events if 'module[classicpilot_nn_impl.dynamic.mat_mul_v3]' in e['text'] and 'result[success]' in e['text']]
launches=[e for e in events if 'StreamLaunchKernelV1:' in e['text'] and 'te_matmulv3_' in e['text']]
assert compiles and launches
kernels={re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0] for e in compiles}
assert all(re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0] in kernels for e in launches)
for e in compiles+launches:
 raw=Path(e['file']);e['raw_retained']=raw.exists() and e['text'] in raw.read_text(errors='replace')
import shutil
binary_dir=out/'native_binary';binary_dir.mkdir(exist_ok=True)
for k in kernels:
 for source in (out/'.cache/all-pairs/ops_nn/graph_cache').rglob(k+'.o'):
  meta=source.with_suffix('.json');info=json.loads(meta.read_text());assert info['sha256']==sha(source)
  shutil.copy2(source,binary_dir/source.name);shutil.copy2(meta,binary_dir/meta.name)
binaries={str(p):sha(p) for p in binary_dir.glob('*.o')};assert binaries
receipts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and (p.suffix in ('.pt','.json','.jsonl','.log','.o') or p.name=='guard.log')}
a={'case_id':c['case_id'],'numerical_pass':True,'registered':False,'parent_accepted':False,'verification':c['verification'],'input_seed':c['input_seed'],'case_snapshot':c['case_snapshot'],'guard':g,'source_pin_checked':True,'original_build_file_pairs_checked':len(compares),'native_compiles':compiles,'native_launches':launches,'binary_sha256':binaries,'receipts':receipts,'checker':str(out/'checker.json'),'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'FROZEN.json').write_text(json.dumps(a,indent=2));(out/'FROZEN.json').chmod(0o444)
print(c['case_id'],'frozen',len(binaries),'binaries')
