import pathlib,json,hashlib
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-20261026/transformer_attention';O=W/'.cache/all-pairs/resume22_server2_transformer_attention01';S=O/'native_build/attention_vendor/src/ops-transformer';D=json.loads((P/'VENDOR.json').read_text()); failures=[];checked=0
for f in D['files']:
 p=pathlib.Path(f['copy'])
 if p.is_file():
  checked+=1
  if hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:failures.append(str(p))
 else:failures.append(str(p))
source_compare=[]
for relative,expected in json.loads((P/'LOCAL_ORIGINAL_HASHES.json').read_text()).items():
 p=S/relative
 source_compare.append({'path':str(p),'matches_current_original':p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,'local_original_sha256':expected})
result={'vendor_receipt_file_count_checked':checked,'changed_or_missing_vendor_files':failures,'original_attention_source_comparison':source_compare,'api':'FlashAttentionScore via aclopCompileAndExecute','adapter':'transformer_graph_gqa','top_level_adaptation':'BSND contiguous inputs; keep_prob=1; scale from canonical params or D**-0.5; no mask for single last-position query with full context; select complete attention output','required_outputs':['o'],'final_state_or_cache_outputs':[],'source_unchanged':bool(source_compare) and all(x['matches_current_original'] for x in source_compare)}
(P/'SOURCE_EVIDENCE.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='original_attention_source_comparison'}))
