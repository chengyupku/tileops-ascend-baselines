import sys,os,json,runpy,time
from pathlib import Path
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');out=P/'all_siblings';out.mkdir(exist_ok=True)
(out/'guard_snapshot.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text())
vendor=P/'gemm/private_provider';os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor);os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']=str(vendor)
for name in ['ASCEND_CACHE_PATH','ASCEND_WORK_PATH','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR']:
 d=out/name;d.mkdir(exist_ok=True);os.environ[name]=str(d)
os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W),str(P)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
from integration700_transformer_dense_quant import quant
from integration700_transformer_dense_fp8_bmm import fp8_bmm
from integration700_transformer_dense_fp8_moe import fp8_moe
factories={'GemmINT8W8A8FwdOp':quant,'BmmFP8FwdOp':fp8_bmm,'MoEGroupedGemmFP8FwdOp':fp8_moe}
prior={('GemmINT8W8A8FwdOp','llama3-70b-dec-qkv-tsr-bfloat16'),('BmmFP8FwdOp','square-1k-float8_e4m3fn-bfloat16'),('MoEGroupedGemmFP8FwdOp','qwen3-235b-dec-b64-down')}
rows=[x for x in json.loads((P/'SCOPE.json').read_text()) if x['operator'] in factories and (x['operator'],x['case_id']) not in prior]
rows.sort(key=lambda x:(list(factories).index(x['operator']),'prefill' in x['case_id'],x['case_id']))
t=time.monotonic();failed=set();records=[]
for row in rows:
 op,cid=row['operator'],row['case_id'];dest=out/(op+'__'+cid+'.json')
 if dest.exists() and json.loads(dest.read_text()).get('status')=='verified':continue
 if op in failed:continue
 if op=='BmmFP8FwdOp' and cid.endswith('-float16'):continue
 if time.monotonic()-t>200:break
 PROVIDERS[op]=factories[op];sys.argv=[str(W/'TileOPs/scripts/ascend_pair_check.py'),'--op',op,'--case-id',cid,'--library','ops-transformer','--output',str(dest)]
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 except SystemExit as e:
  if e.code:failed.add(op)
 records.append(dict(op=op,case_id=cid,status=json.loads(dest.read_text()).get('status') if dest.exists() else 'no_receipt'))
(out/'BATCH.json').write_text(json.dumps(dict(rows=records,wall_s=time.monotonic()-t,stopped_families=sorted(failed)),indent=2))
