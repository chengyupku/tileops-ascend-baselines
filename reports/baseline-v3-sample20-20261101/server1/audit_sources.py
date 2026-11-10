import pathlib,json
P=pathlib.Path(__file__).resolve().parent
old=json.loads((P/'HISTORICAL_CORRECTNESS.json').read_text());results=json.loads((P/'RESULTS.json').read_text());sample=json.loads((P/'SAMPLE.json').read_text());bykey={tuple(r['key']):r for r in sample['rows']};out=[]
for row in results['rows']:
 p=pathlib.Path(row['receipt']);r=json.loads((p.parent/'correctness.json').read_text());tid=bykey[tuple(row['key'])]['task_id'];o=old[tid]
 x=o['source_sha256'];y=r['source_sha256'];both=x.keys()&y.keys();vals=set(y.values());diff=[k for k in both if x[k]!=y[k]];unmatched=[k for k,h in x.items() if h not in vals and k.endswith(('.py','.cpp','.asc'))]
 # Relocated launchers only select same original provider/runtime and v3 timing.
 permitted={'tools/baseline_retest32_cpu_conv_v26_entry.py','tools/baseline_retest32_w4_original_v28_entry.py','tools/baseline_retest32_tilelang_v46_entry.py','tools/baseline_retest32_tilelang_v47_entry.py'}
 # Existing GDN path and its source are unchanged; additions only register other APIs.
 if row['key'][0]=='vllm-ascend' and row['key'][1]=='GDNFwdOp':permitted.update({'TileOPs/benchmarks/ascend_all_pairs/vllm/contracts.py','TileOPs/benchmarks/ascend_all_pairs/vllm/__init__.py'})
 item={'key':row['key'],'api_equal':o['api']==r['api'],'adaptation_equal':o.get('adaptation')==r.get('adaptation'),'seed_equal':o['input_seed']==r['input_seed'],'case_equal':o['case_snapshot']==r['case_snapshot'],'old_source_count':len(x),'old_hashes_present_in_new':sum(h in vals for h in x.values()),'unmatched_python_cpp':unmatched,'unexplained_python_cpp':[k for k in unmatched if k not in permitted],'same_path_changed':diff}
 out.append(item)
(P/'FINAL_SOURCE_AUDIT.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
