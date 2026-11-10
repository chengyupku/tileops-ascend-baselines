from pathlib import Path
import json,shutil,hashlib,math,ast,subprocess,difflib
R=Path('/opt/tileops-all-pairs/workspace'); W=R/'reports/baseline-integration-20261026/transformer_gdn';T=R/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks';S=T/'recovery0730-gdn-q35-small-abi';F=T/'recovery0730-gdn-fixed-abi';OWNER='selected-integration-20261026/transformer_gdn'
scope=json.loads((W/'SCOPE.json').read_text());ids={x['case_id'] for x in scope};keys={tuple(x['key']) for x in scope}
def write(n,x):(W/n).write_text(json.dumps(x,indent=2))
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
contracts={r['case_id']:r for r in json.loads((R/'TileOPs/benchmarks/ascend_all_pairs/vllm/inventory_snapshot.json').read_text())['cases'] if ( 'ops-transformer',r['op'],r['case_id']) in keys};assert len(contracts)==len(scope)
write('CONTRACTS.json',contracts)
history=[]
for case in scope:
 for task in [S,F,T/'recovery0730-gdn-q35-large-abi',T/'recovery0730-gdn-varlen-symmetric-abi',T/'recovery0730-delta-whole-api-deployment']:
  d=task/case['case_id']
  for n in ['checker.json','GUARD_RECEIPT.json','fresh_resource_admission.json']:
   if (d/n).exists():history.append({'key':case['key'],'evidence':pin(d/n),'data':json.loads((d/n).read_text())})
amnesty=T.parent.parent/'tasks/amnesty-transformer-gdn128-cpu/EXACT_DISPOSITION.json'
write('HISTORY.json',{'receipts':history,'amnesty':[x for x in json.loads(amnesty.read_text())['rows'] if tuple(x['key']) in keys],'amnesty_source':pin(amnesty)})
assert not any(x['data'].get('status')=='verified' and x['data'].get('op')==x['key'][1] and x['data'].get('case_id')==x['key'][2] for x in history)
P=W/'chunk_package';P.mkdir(exist_ok=True)
for f in (S/'chunk_package').iterdir():
 if f.is_file():shutil.copyfile(f,P/f.name)
r=json.loads((P/'RUNTIME_RECIPE.json').read_text());r['required_environment']['GDN_CASE_OWNER']=OWNER;r['guard']['owner']=OWNER;write('chunk_package/RUNTIME_RECIPE.json',r)
s=(P/'binding.py').read_text();orig=s
s=s.replace("else initial.transpose(-2,-1).contiguous().clone()","else (initial if p.get('state_v_first',False) else initial.transpose(-2,-1)).contiguous().clone()")
s=s.replace('final.transpose(-2,-1).contiguous()','(final if p.get(\'state_v_first\',False) else final.transpose(-2,-1).contiguous())')
s=s.replace('g.reshape(-1,hv).float().contiguous()','None if g is None else g.reshape(-1,hv).float().contiguous()')
s=s.replace("return Candidate(Implementation(run)","\n    if c['op']=='DeltaNetFwdOp':\n        gdn_run=run\n        def run(q,k,v,beta,initial,offsets,offsets_cpu):\n            return gdn_run(q,k,v,None,beta,initial,offsets,offsets_cpu,None,None)\n    return Candidate(Implementation(run)")
(P/'binding.py').write_text(s);ast.parse(s)
(W/'adapter.patch').write_text(''.join(difflib.unified_diff(orig.splitlines(True),s.splitlines(True),fromfile=str(S/'chunk_package/binding.py'),tofile='chunk_package/binding.py')))
s=(P/'entry.py').read_text().replace("== 'GDNFwdOp'","in ('GDNFwdOp','DeltaNetFwdOp')");(P/'entry.py').write_text(s)
est={};chunks=[]
for i,c in contracts.items():
 p=c['params'];assert not any(p.get(f,False) for f in ['use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval','use_gate_in_kernel'])
 n=sum(math.prod(v[0])*{'bfloat16':2,'float32':4,'int64':8}[v[1]] for v in p.values() if isinstance(v,list))
 est[i]={'declared_tensor_bytes':n,'minimum_effective_available_host_bytes':40*2**30+max(0,12*n-4*2**30),'minimum_free_disk_bytes':82*2**30,'estimated_device_budget_bytes':2*2**30+8*n}
 d=W/i;d.mkdir(exist_ok=True)
 if i=='q35-gva-decode-b512-bfloat16':continue
 chunks.append(i)
 child=(S/'q35-27b-8192-bfloat16/child.sh').read_text().replace(str(S/'q35-27b-8192-bfloat16'),str(d)).replace('cd "'+str(S)+'"','cd "'+str(W)+'"').replace(str(S/'chunk_package/entry.py'),str(P/'entry.py')).replace('regional-server2/continuation0430/'+S.name,OWNER).replace('--case-id q35-27b-8192-bfloat16','--case-id '+i).replace('--op GDNFwdOp','--op '+c['op'])
 for line in child.splitlines():
  if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
 (d/'child.sh').write_text(child);assert subprocess.run(['bash','-n',str(d/'child.sh')]).returncode==0
write('CHUNK_CASES.json',sorted(chunks,key=lambda i:est[i]['declared_tensor_bytes']));write('CASE_RESOURCE_ESTIMATES.json',est)
# Decode reuses the entire previously verified full-batch RecurrentKda binding.
P2=W/'recurrent_package';P2.mkdir(exist_ok=True)
for f in (F/'package').iterdir():
 if f.is_file():shutil.copyfile(f,P2/f.name)
r=json.loads((P2/'RUNTIME_RECIPE.json').read_text());r['required_environment']['GDN_CASE_OWNER']=OWNER;r['guard']['owner']=OWNER;write('recurrent_package/RUNTIME_RECIPE.json',r)
write('RECURRENT_CASES.json',['q35-gva-decode-b512-bfloat16'])
i='q35-gva-decode-b512-bfloat16';d=W/i
child=(F/i/'child.sh').read_text().replace(str(F/i),str(d)).replace('cd "'+str(F)+'"','cd "'+str(W)+'"').replace(str(F/'package/entry.py'),str(P2/'entry.py')).replace('regional-server2/continuation0430/'+F.name,OWNER)
for line in child.splitlines():
 if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
(d/'child.sh').write_text(child)
pins=json.loads((F/'SOURCE_EVIDENCE.json').read_text())['pins'];write('SOURCE_RECHECK.json',{'pins':pins,'changed':[x for x in pins if pin(Path(x['path']))['sha256']!=x['sha256']]})
write('CHECKS.json',{'scope_count':len(scope),'contracts':len(contracts),'binding_ast':True,'shell_syntax':True,'checker':pin(Path(r['checker']))})
print('prepared',len(contracts));print('source pins changed',json.loads((W/'SOURCE_RECHECK.json').read_text())['changed'])
