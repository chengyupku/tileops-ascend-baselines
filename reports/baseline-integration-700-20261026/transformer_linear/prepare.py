from pathlib import Path
import json,shutil,math,ast
R=Path('/opt/tileops-all-pairs/workspace');W=R/'reports/baseline-integration-700-20261026/transformer_linear';O=R/'reports/baseline-integration-20261026/transformer_gdn';OWNER='baseline700-20261026/transformer_linear'
scope=json.loads((W/'SCOPE.json').read_text());keys={(x['operator'],x['case_id']) for x in scope}
inv=json.loads((R/'TileOPs/benchmarks/ascend_all_pairs/vllm/inventory_snapshot.json').read_text())['cases']; cs=[x for x in inv if (x['op'],x['case_id']) in keys];assert len(cs)==109
(W/'ALL_CONTRACTS.json').write_text(json.dumps(cs,indent=2));gd={x['case_id']:x for x in cs if x['op']=='GDNFwdOp'};(W/'CONTRACTS.json').write_text(json.dumps(gd,indent=2))
P=W/'chunk_package';P.mkdir(exist_ok=True)
for f in (O/'chunk_package').iterdir():
 if f.is_file():shutil.copyfile(f,P/f.name)
r=json.loads((P/'RUNTIME_RECIPE.json').read_text());r['required_environment']['GDN_CASE_OWNER']=OWNER;r['guard']['owner']=OWNER;(P/'RUNTIME_RECIPE.json').write_text(json.dumps(r,indent=2))
ids=[i for i,c in gd.items() if c['params']['q'][1]=='bfloat16' and c['params']['q'][0][1]>1];(W/'CHUNK_CASES.json').write_text(json.dumps(ids))
est={}
for i in ids:
 c=gd[i];p=c['params'];assert not any(p.get(f,False) for f in ['use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval','use_gate_in_kernel'])
 n=sum(math.prod(v[0])*{'bfloat16':2,'float32':4,'int64':8}[v[1]] for v in p.values() if isinstance(v,list))
 est[i]={'declared_tensor_bytes':n,'minimum_effective_available_host_bytes':40*2**30+max(0,12*n-4*2**30),'minimum_free_disk_bytes':82*2**30,'estimated_device_budget_bytes':2*2**30+8*n}
 d=W/i;d.mkdir(exist_ok=True)
 child=(O/'q35-27b-1024-7168-bfloat16/child.sh').read_text().replace(str(O/'q35-27b-1024-7168-bfloat16'),str(d)).replace(str(O),str(W)).replace('selected-integration-20261026/transformer_gdn',OWNER).replace('--case-id q35-27b-1024-7168-bfloat16','--case-id '+i)
 for line in child.splitlines():
  if line.startswith('export ') and '="'+str(d)+'/' in line:Path(line.split('="')[1].rstrip('"')).mkdir(exist_ok=True)
 (d/'child.sh').write_text(child)
(W/'CASE_RESOURCE_ESTIMATES.json').write_text(json.dumps(est,indent=2));print('prepared',len(ids))
