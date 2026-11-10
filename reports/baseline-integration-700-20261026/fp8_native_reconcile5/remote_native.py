import pathlib,json,hashlib,subprocess
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense';v=P/'gemm/private_provider'
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
configs=[]
for f in (v/'op_impl/ai_core/tbe/kernel/config/ascend950').glob('*.json'):
 if 'grouped' in f.name.lower():configs.append(dict(**ref(f),data=json.loads(f.read_text())))
root=W/'ascend_baselines/ops-transformer/gmm/grouped_matmul/op_api';sources=[]
for name in ['aclnn_grouped_matmul.cpp','grouped_matmul.cpp']:
 f=root/name;sources.append(dict(path=str(f),exists=f.exists()))
cache=P/'fp8_siblings01/ASCEND_CACHE_PATH';cachefiles=[str(f.relative_to(cache)) for f in cache.rglob('*') if f.is_file()]
lib=v/'op_api/lib/libcust_opapi.so';nm=subprocess.run(['nm','-D','--defined-only',str(lib)],capture_output=True,text=True,check=True)
print(json.dumps(dict(configs=configs,sources=sources,cache=dict(path=str(cache),exists=cache.exists(),files=cachefiles),native_symbols=[x for x in nm.stdout.splitlines() if 'aclnnGroupedMatmulV5' in x],library=ref(lib))))
