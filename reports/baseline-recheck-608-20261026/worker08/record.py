import json,datetime,pathlib,yaml
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=ROOT/'reports/baseline-recheck-608-20261026/worker08'
rows=json.load(open(OUT/'SCOPE.json'))['rows']
specs={}
for p in (ROOT/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
 for k,v in (yaml.safe_load(p.read_text()) or {}).items():
  if isinstance(v,dict): specs[k]=(p,v)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def contract(r):
 p,s=specs[r['operator']]; ws=[w for w in s.get('workloads',[]) if r['case_id']==w.get('label') or r['case_id'].startswith(w.get('label','!')+'-')]; w=max(ws,key=lambda w:len(w['label'])) if ws else {}
 return {'case_id':r['case_id'],'spec':str(p)+':'+r['operator'],'signature':s.get('signature',{}),'workload':w}
def add(b,o,fn,ev,api):
 for r in rows:
  if (r['baseline'],r['operator'])!=(b,o):continue
  d,reason=fn(r,contract(r))
  rec={'key':r['key'],'decision':d,'reason':reason,'evidence':[str(ROOT/e) if not e.startswith('/') else e for e in ev]+[contract(r)['spec']], 'contract':contract(r),'api_or_rejected_routes':api,'decided_at':now()}
  with open(OUT/'decisions.jsonl','a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
 n=sum(1 for _ in open(OUT/'decisions.jsonl'))
 (OUT/'HEARTBEAT.json').write_text(json.dumps({'at':now(),'family':[b,o],'completed':n,'blockers':[]},ensure_ascii=False))
if __name__=='__main__':
 add('tilelang','Conv2dFwdOp',lambda r,c:('不要接','resnet-3x3 BF16合同；convolution及autotune入口均在内部覆盖dtype为float16，BF16转FP16非无损且无FP32入口。') if 'bfloat16' in r['case_id'] else ('要接',f"{r['case_id']}，{c['workload']}：NCHW/HWCF转置，原版convolution的S/P/D参数及im2col+GEMM完整实现该无bias卷积；FP16/FP32累加，尾块由原版ceildiv处理。"),['ascend_baselines/tilelang/examples/convolution/example_convolution.py:19','ascend_baselines/tilelang/examples/convolution/example_convolution_autotune.py:84'],'convolution; convolution_autotune')
 add('tilelang-ascend','GemmFwdOp',lambda r,c:('要接',f"{r['case_id']}：{c['workload']}；matmul的dtype直接传bfloat16，B从[N,K]转置连续化；M小于128可零填充至128，N=2112可选block_N=64或补零裁剪；K均64整除，原版kernel完成全K累加并直接写BF16。"),['ascend_baselines/tilelang-ascend/examples/gemm/example_gemm.py:21'],'matmul')
 add('ops-nn','GroupedGemmFwdOp',lambda r,c:('要接',f"{r['case_id']}：M4096/G16/N4096/K4096，按batch_sizes/offsets切16个独立[256,4096]矩阵，按transpose_b转置每组B；aclnnMm原版支持该FP16/BF16输入和同dtype输出，逐组结果恢复[M,N]，无额外数学。"),['ascend_baselines/ops-nn/matmul/mat_mul_v3/docs/aclnnMm.md:26'],'aclnnMmGetWorkspaceSize/aclnnMm')
 add('ops-math','FFTC2CFwdOp',lambda r,c:('要接',f"{r['case_id']}：{c['workload']}；aclStft支持complex64/128输入输出，设置nFft=winLength=L=n、hopLength=n、normalized=false、onesided=false、returnComplex=true、window=null得到单窗口完整DFT，去掉frames=1轴即可。复数走原版AiCPU完整分支，不需要自行分解实虚或FFT。"),['ascend_baselines/ops-math/math/stft/docs/aclStft.md:26','ascend_baselines/ops-math/math/stft/op_host/op_api/acl_stft.cpp:CheckShape','ascend_baselines/ops-math/math/stft/op_host/op_api/stft.cpp:StftAiCpu','ascend_baselines/ops-math/math/stft/op_host/op_api/stft.cpp:GetOutputTypeByInput'],'aclStftGetWorkspaceSize/aclStft (complex AiCPU branch)')
