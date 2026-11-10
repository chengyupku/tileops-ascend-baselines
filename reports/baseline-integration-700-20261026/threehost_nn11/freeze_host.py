from pathlib import Path
import json,hashlib,sys,re,shutil,datetime
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace') if str(P).startswith('/opt/') else Path('/root/dyq/workspace');out=P/sys.argv[1]
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((out/'checker.json').read_text());g=json.loads((out/'GUARD_RESULT.json').read_text())
assert g['returncode']==0 and c['status']=='verified'
assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
assert c['independent_cpu_reference'] and c['reference_device']=='cpu'
assert c['input_seed']==int.from_bytes(hashlib.sha256(f"GroupedGemmFwdOp::{c['case_id']}".encode()).digest()[:4],'big')
source_checks={}
for k,v in c['source_sha256'].items():
 f=P/'snapshot'/k if k=='TileOPs/workloads/numerics.py' else W/k
 assert f.exists() and h(f)==v,(k,str(f));source_checks[str(f)]=v
for k,v in c['oracle_source_sha256'].items():
 f=P/'snapshot'/k;assert h(f)==v,k;source_checks[str(f)]=v
symbols=[json.loads(x) for x in (out/'ACTUAL_SYMBOLS.jsonl').read_text().splitlines()]
assert symbols
for s in symbols:
 assert s['pointer'] and s['maps']
 for m in s['maps']:
  f=Path(m['path']);assert f.stat().st_ino==m['inode'] and h(f)==m['sha256'];lo,hi=[int(x,16) for x in m['line'].split()[0].split('-')];assert lo<=s['pointer']<hi
native=[json.loads(x) for x in (out/'native.jsonl').read_text().splitlines()]
compiles=[e for e in native if '.dynamic.mat_mul_v3]' in e['text'] and 'result[success]' in e['text']]
launches=[e for e in native if 'StreamLaunchKernelV1:' in e['text'] and 'te_matmulv3_' in e['text']]
assert compiles and launches
names={re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0] for e in compiles}
assert all(re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0] in names for e in launches)
bindir=out/'native_binary';bindir.mkdir(exist_ok=True)
for name in names:
 for f in (out/'.cache/all-pairs/ops_nn/graph_cache').rglob(name+'.o'):
  meta=f.with_suffix('.json');info=json.loads(meta.read_text());assert info['sha256']==h(f)
  shutil.copy2(f,bindir/f.name);shutil.copy2(meta,bindir/meta.name)
assert list(bindir.glob('*.o'))
deps={}
for line in (out/'candidate_maps.txt').read_text().splitlines():
 a=line.split()
 if len(a)>=6 and a[-1].startswith('/'):
  f=Path(a[-1])
  if f.is_file() and ('.so' in f.name or '/bin/' in str(f)):
   st=f.stat();deps[str(f)]=dict(sha256=h(f),inode=st.st_ino,device=st.st_dev)
receipts={str(f.relative_to(out)):h(f) for f in out.rglob('*') if f.is_file() and f.name!='FROZEN.json' and (f.suffix in ['.pt','.json','.jsonl','.log','.o','.txt'] or f.name=='guard.log')}
proof=dict(case_id=c['case_id'],status='verified_pending_parent_review',accepted_added=0,registered=False,guard=g,verification=c['verification'],case_snapshot=c['case_snapshot'],input_seed=c['input_seed'],source_sha256=source_checks,actual_symbols=symbols,loaded_native_dependencies=deps,native_compiles=compiles,native_launches=launches,binary_sha256={str(f):h(f) for f in bindir.glob('*.o')},artifact_sha256=receipts,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'FROZEN.json').write_text(json.dumps(proof,indent=2));print('verified frozen',c['case_id'],len(compiles),'compiles',len(launches),'launches',len(deps),'native dependencies')
