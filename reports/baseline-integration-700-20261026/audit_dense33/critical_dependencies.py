import pathlib,json,hashlib,sys
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense';SCOPE=SCOPE_PLACEHOLDER
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=[]
for r in SCOPE:
 c=json.loads((P/r['checker_receipt']).read_text());checks=[]
 for rel,expected in [('TileOPs/scripts/ascend_pair_check.py',c['verifier_sha256']),('TileOPs/benchmarks/ascend_baselines.py',c['adapter_sha256'])]:
  p=W/rel;checks.append(dict(path=str(p),sha256=sha(p),expected=expected,matches=sha(p)==expected))
 if r['operator'] in ['MoEPrePermuteFwdOp','YaRNFwdOp']:
  rel='TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_dense_graph.py';p=W/rel;expected=c['provider_source_sha256'].get(rel,sha(P/'integration700_transformer_dense_graph.py'));checks.append(dict(path=str(p),sha256=sha(p),expected=expected,matches=sha(p)==expected,comparison='receipt provider hash' if rel in c['provider_source_sha256'] else 'private tested binding file, prior parent reviewed GraphCall'))
 out.append(dict(key=[r['baseline'],r['operator'],r['case_id']],checks=checks))
p=P/'fp8_siblings01/ASCEND_PROCESS_LOG_PATH';inventory=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in p.rglob('*') if f.is_file()];remaining=[]
for f in p.rglob('*'):
 if not f.is_file() or '/debug/plog/' in str(f):continue
 hits=[dict(line=i,text=l) for i,l in enumerate(f.read_text(errors='replace').splitlines(),1) if any(t in l.lower() for t in ['groupedmatmul','grouped_matmul','kernel_name'])];remaining.append(dict(path=str(f),hits=hits[:10],count=len(hits)))
print(json.dumps(dict(rows=out,sibling_native_log_inventory=inventory,non_debug_kernel_attribution_search=remaining)))
