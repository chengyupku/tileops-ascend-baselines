from pathlib import Path
import json,hashlib,shlex
W=Path('/opt/tileops-all-pairs/workspace');R=W/'reports/baseline-integration-20261026';P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';D=R/'register_server2/repair01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
manifest={'groups':{},'contracts':{},'sources':{}}
for group,op in [('attention','GQADenseFwdOp'),('gdn','GDNFwdOp')]:
 root=R/('transformer_'+group);cfg={'root':str(root),'pins':{},'environment':{},'directories':[]};manifest['groups'][group]=cfg
 result=json.loads((root/'RESULT.json').read_text());rows=result['rows']
 for row in rows:
  if row['status']!='verified' or row['key'][1]!=op:continue
  cid=row['key'][2];receipt=Path(row.get('checker_receipt',str(root/cid/'checker.json')))
  rec=json.loads(receipt.read_text());assert rec['status']=='verified'
  manifest['contracts'][cid]=dict(group=group,op=op,snapshot=rec['case_snapshot'])
 if group=='gdn':
  chunk=root/'chunk_package';recipe=json.loads((chunk/'RUNTIME_RECIPE.json').read_text());assert recipe['dependency_closure_ready']
  cfg['pins'].update({x['path']:x['sha256'] for x in json.loads((chunk/'RUNTIME_PINS.json').read_text())['files']})
  child=root/'prefill-v-first-narrow-bfloat16/child.sh'
  for line in child.read_text().splitlines():
   if line.startswith('export '):
    for item in shlex.split(line[7:]):
     k,v=item.split('=',1)
     if '/prefill-v-first-narrow-bfloat16' not in v:cfg['environment'][k]=v
  cfg['python_paths']=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),*recipe['python_paths']]
  cfg['cwd']=str(root);cfg['executable']='/opt/tileops-ci/runtime/bin/python'
  for name in ['binding.py','direct_acl.py','resource_gate.py','entry.py','RUNTIME_RECIPE.json','RUNTIME_PINS.json','RESOURCE_ESTIMATES.json']:
   cfg['pins'][str(chunk/name)]=sha(chunk/name)
  for name in ['CONTRACTS.json','CHUNK_CASES.json','CASE_RESOURCE_ESTIMATES.json']:cfg['pins'][str(root/name)]=sha(root/name)
  for name in ['binding','direct_acl','resource_gate']:manifest['sources'][name]=str(chunk/(name+'.py'))
 else:
  vendor=json.loads((root/'VENDOR.json').read_text());cfg['vendor']=vendor['installed_vendor'];cfg['pins'].update(json.loads((root/'SOURCE_PINS.json').read_text()))
  for x in vendor['files']:
   # Manifest formats are checked explicitly below.
   pass
  cfg['environment']={'TRANSFORMER_ATTENTION_VENDOR':cfg['vendor'],'ASCEND_CUSTOM_OPP_PATH':cfg['vendor'],'ASCEND_HOME_PATH':'/usr/local/Ascend/cann-9.1.0','ASCEND_OPP_PATH':'/usr/local/Ascend/cann-9.1.0/opp','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','FLA_NPU_DISABLE_PTH':'1','PYTHONDONTWRITEBYTECODE':'1','ASCEND_GLOBAL_LOG_LEVEL':'0'}
  cfg['python_paths']=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(W/'.cache/all-pairs/resume22_server2_transformer_attention01/native_build/runtime_deps/site-packages'),'/opt/tileops-ci/runtime/lib/python3.12/site-packages']
  cfg['executable']='/opt/tileops-all-pairs/bin/allpairs-python'
  for name in ['ascend_graph_call','ascend_transformer_graph','ascend_transform_baselines']:
   f=root/(name+'.py');manifest['sources'][name]=str(f);cfg['pins'][str(f)]=sha(f)
  cfg['pins'][str(root/'VENDOR.json')]=sha(root/'VENDOR.json')
  # Pin every installed vendor file and its verified source closure against receipt hashes.
  cfg['pins'].update(vendor['source_closure'])
  for x in vendor['files']+vendor['added_unchanged_transitive_headers']:
   for field in ['source','copy']:cfg['pins'][x[field]]=x['sha256']
  cfg['directories'].append(cfg['vendor'])
 cfg['directories']+=cfg['python_paths']+[v for k,v in cfg['environment'].items() if k in ('ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH')]
 if 'LD_LIBRARY_PATH' in cfg['environment']:cfg['directories']+=cfg['environment']['LD_LIBRARY_PATH'].split(':')
 cfg['pins'][str(root/'RESULT.json')]=sha(root/'RESULT.json')
dump(D/'manifest_draft.json',manifest)
print('contracts',len(manifest['contracts']))
