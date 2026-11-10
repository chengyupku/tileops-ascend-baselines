import json,datetime,pathlib,yaml
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=ROOT/'reports/baseline-recheck-608-20261026/worker05'
rows=json.loads((OUT/'SCOPE.json').read_text())['rows']
def record(base,op,decision,reason,evidence,route):
 spec=None; sp=None
 for f in (ROOT/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
  d=yaml.safe_load(f.read_text())
  if isinstance(d,dict) and op in d: spec=d[op];sp=f;break
 assert spec is not None,op
 for row in rows:
  if (row['baseline'],row['operator'])!=(base,op):continue
  matches=[w for w in spec.get('workloads',[]) if row['case_id'].startswith(str(w.get('label','!!'))+'-') or row['case_id']==w.get('label')]
  matches.sort(key=lambda w:len(str(w.get('label'))),reverse=True)
  contract={'signature':spec.get('signature'),'workload':matches[0] if matches else None,'case_id':row['case_id']}
  item={'key':row['key'],'decision':decision,'reason':row['case_id']+'：'+reason,'evidence':[str(ROOT/e) if not e.startswith('/') else e for e in evidence]+[str(sp)+':'+op],'contract':contract,'api_or_rejected_routes':route,'decided_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  with (OUT/'decisions.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 n=len((OUT/'decisions.jsonl').read_text().splitlines());(OUT/'HEARTBEAT.json').write_text(json.dumps({'current_family':[base,op],'completed':n,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'blockers':[]},ensure_ascii=False))
if __name__=='__main__':
 record('tilelang-ascend','BmmFwdOp','要接','原版batch_matmul完整实现三维A@B，输入输出dtype可传float16/bfloat16、FP32累加；该case的M/N/K均被block_M=128、block_N=128、K_L1=64整除，直接传B/M/N/K及dtype即可，无额外数学。',['ascend_baselines/tilelang-ascend/examples/batch_gemm/batch_gemm.py:28'],'batch_matmul')
 record('tilelang-ascend','AbsFwdOp','要接','16384×16384 FP16符合vec_abs完整搬入、abs、写回；128×256块整除，直接dtype=float16。',['ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py:64'],'vec_abs')
 record('tilelang-ascend','LogFwdOp','要接','16384×16384 BF16可无损扩宽FP32后调用ln(dtype=float)，完整执行自然对数并写回，再按合同转回BF16；128×256块整除。',['ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_elementwise.py:2856'],'ln')
