"""CPU-only, exact-scope deployment audit. Does not import device runtimes."""
from pathlib import Path
import json, hashlib, difflib, subprocess, os, socket, datetime
W=Path('/root/dyq/workspace'); D=Path(__file__).resolve().parent
R=W/'reports/baseline-integration-20261026'; C=W/'.cache/all-pairs/ops_transformer'
scope=json.loads((D/'SCOPE.json').read_text()); keys={tuple(r['key']) for r in scope}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v): (D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def pin(p): return {'path':str(p),'sha256':sha(p),'size':p.stat().st_size}
def compare(a,b):
 rows=[]
 for p in sorted(a.rglob('*')):
  q=b/p.relative_to(a)
  if p.is_file() and q.is_file(): rows.append({'relative':str(p.relative_to(a)),'source_sha256':sha(p),'deployed_sha256':sha(q),'equal':sha(p)==sha(q)})
 return {'source':str(a),'target':str(b),'compared':len(rows),'differences':[r for r in rows if not r['equal']], 'files':rows}
V=C/'install/vendors/allpairs_transformer'; E=C/'extracted/packages/vendors/allpairs_transformer'; K=V/'op_impl/ai_core/tbe/allpairs_transformer_impl/ascendc'
wrapper='op_impl/ai_core/tbe/allpairs_transformer_impl/dynamic/'
diffs=[]
for name in ['grouped_matmul_apt.py','grouped_matmul.py']:
 a=E/wrapper/name;b=V/wrapper/name
 text=''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile=str(a),tofile=str(b)))
 (D/(name+'.deployed.diff')).write_text(text)
 diffs.append({'original':pin(a),'installed':pin(b),'diff':name+'.deployed.diff','change':'Removed original -Werror from custom_compile_options; not a top-level parameter/layout adaptation.'})
source=W/'ascend_baselines/ops-transformer'
comparisons={
 'grouped_source_vs_build':compare(source/'gmm/grouped_matmul',C/'source/gmm/grouped_matmul'),
 'grouped_source_vs_installed':compare(source/'gmm/grouped_matmul',K/'grouped_matmul'),
 'tensor_api_vs_installed':compare(W/'ascend_baselines/ops-tensor/include/tensor_api',K/'common/tensor_api'),
 'package_vs_installed':compare(E,V)}
dump('SOURCE_COMPARISONS.json',comparisons)
# Retain compact first-hand excerpts, rather than copy all historical logs.
log=C/'logs/debug/plog/plog-2596243_20261022110923932.log'
lines=log.read_text().splitlines();excerpt=[{'line':i+1,'text':t} for i,t in enumerate(lines) if 4495<=i+1<=4520]
dump('LOCAL_JIT_HISTORY.json',{'file':pin(log),'excerpts':excerpt,'interpretation':'Nonquant local compile succeeded using installed modified wrapper; not original numerical validation.'})
layout=K/'common/tensor_api/include/tensor_api/tensor/layout_type.h'
config=V/'op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'
elves=[V/'op_api/lib/libcust_opapi.so',V/'op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so',V/'op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so']
symbols=subprocess.check_output(['nm','-D','--defined-only',str(elves[0])],text=True)
original_flags=source/'gmm/grouped_matmul/op_host/CMakeLists.txt'
evidence={'host':socket.gethostname(),'architecture':os.uname().machine,'package':pin(C/'source/build_out/cann-ops-transformer-allpairs_linux-x86_64.run'),'build_cache':pin(C/'source/build/CMakeCache.txt'),'source_git_head':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'source_git_status':subprocess.check_output(['git','-C',str(source),'status','--short'],text=True),'wrapper_mutations':diffs,'original_flags':{'file':pin(original_flags),'text':original_flags.read_text()},'elfs':[dict(pin(p),elf_machine=int.from_bytes(p.read_bytes()[18:20],'little')) for p in elves],'api_exports':[s for s in ['aclnnGroupedMatmulV5','aclnnGroupedMatmulV5GetWorkspaceSize'] if any(t.split()[-1]==s for t in symbols.splitlines() if t.split())],'config':{'file':pin(config),'GroupedMatmul':json.loads(config.read_text()).get('GroupedMatmul')},'deployed_device_object_count':len(list(V.rglob('*.o'))),'device_delivery':'JIT source; no package .o files. Compiled local cache is not accepted as original because wrapper was changed.','tensor_api':{'file':pin(layout),'namespace_lines':[{'line':i+1,'text':t} for i,t in enumerate(layout.read_text().splitlines()) if 'namespace' in t]},'server2_blocker_applies_to_local':False,'reason':'Local bundled tensor API declares asc::te and differs from server2. Terminal local issue is modified compiler wrapper, independently evidenced.','terminal_disposition':'blocked_modified_original_compile_wrapper','device_requests':0,'checker_run':False}
dump('DEPLOYMENT_EVIDENCE.json',evidence)
# Only retain results for exact keys; no historical success re-certification.
receipts=[]
for name in ['local_verified.json','server1_verified.json','server2_verified.json']:
 d=json.loads((R/name).read_text());receipts += [dict(x,snapshot=name) for x in d['receipts'] if tuple(x.get('key',[])) in keys]
scanned=0
for base,dirs,names in os.walk(W/'.cache/all-pairs/full_rollout_20261024'):
 dirs[:]=[x for x in dirs if x not in ['.git','__pycache__','cache','torch-cache','build','artifacts','tmp','site-packages']]
 for name in names:
  if not (name=='RESULT.json' or ('checker' in name and name.endswith('.json'))):continue
  p=Path(base)/name;scanned+=1
  try:d=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if not isinstance(d,dict):continue
  k=tuple(d.get(x) for x in ('library','op','case_id'))
  if k in keys and d.get('status')=='verified':receipts.append({'key':list(k),'path':str(p),'kind':'fresh_local_checker'})
  for row in d.get('rows',[]) if isinstance(d.get('rows'),list) else []:
   if isinstance(row,dict) and tuple(row.get('key',[])) in keys and row.get('status')=='verified':receipts.append({'key':row['key'],'path':str(p),'kind':'fresh_local_result'})
policy=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/user_amnesty_0323/POLICY.json'
amnesty=[k for k in json.loads(policy.read_text())['amnesty_keys'] if tuple(k) in keys]
target=W/'reports/baseline-status-20261026/EXACT_TARGET_SETS.snapshot.jsonl'
rows=[json.loads(t) for t in target.read_text().splitlines() if t.strip()];matched=[x for x in rows if tuple(x.get('key',[])) in keys]
dump('DEDUP.json',{'scope_count':len(keys),'verified_matches':receipts,'fresh_local_receipt_files_scanned':scanned,'remote_read_scope':'Latest collected server1/server2 snapshots only; no remote queries requested or performed.','amnesty_policy':pin(policy),'amnesty_matches':amnesty,'target_snapshot':pin(target),'target_matches':[{'key':x['key'],'classification':x.get('classification'),'substatus':x.get('substatus'),'historical_success_claim':x.get('historical_success_claim')} for x in matched]})
print(json.dumps({'source_comparisons':{k:{'compared':v['compared'],'differences':len(v['differences'])} for k,v in comparisons.items()},'modified_grouped_wrappers':len(diffs),'verified_matches':len(receipts),'amnesty_matches':len(amnesty),'target_matches':len(matched),'fresh_receipts_scanned':scanned}))
