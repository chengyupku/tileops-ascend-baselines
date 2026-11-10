from pathlib import Path
import json,yaml,hashlib,ast,collections
P=Path(__file__).resolve().parent;W=P.parents[2];V=W/'ascend_baselines/tilelang-ascend';spec=yaml.safe_load((P/'frozen/TileOPs/src/tileops/manifest/spec/attention.yaml').read_text());rows=[]
for r in json.loads((P/'SCOPE.json').read_text()):
 op,cid=r['operator'],r['case_id'];dtype=cid.rsplit('-',1)[1];label=cid[:-(len(dtype)+1)];ws=next(x for x in spec[op]['workloads'] if x['label']==label);s=dict(ws);s.pop('dtype_cases');mla=op.startswith('MLA')
 src='examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py' if mla else 'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py';api='sparse_attention_fwd' if mla else 'gqa_sink_fwd'
 tree=ast.parse((V/src).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==api)
 prior='mla-tp-b1-float16' if mla else 'serving-kv512-float16'
 branch=('mla_H128_replicate2' if s['H']==128 else 'mla_H64_replicate1') if mla else ('mha_causal' if s.get('is_causal') else 'mha_noncausal')
 args=[s['H'],s['D'],s['PE'],s['N_kv'],1] if mla else [1,1,s['H'],128,64,((1000 if label=='speculative-decode' else (2048 if 'opt-' in label else s['N_kv']))+127)//128*128,s.get('is_causal',False)]
 rows.append(dict(key=r['key'],manifest=s,dtype=dtype,source=src,source_sha256=hashlib.sha256((V/src).read_bytes()).hexdigest(),builder=api,function_lines=[fn.lineno,fn.end_lineno],decorator=ast.unparse(fn.decorator_list[0]),builder_args=args,builder_kwargs=dict(kv_group=s['H_kv'],dtype=dtype) if mla else dict(sm_scale=s['D']**-.5,mask_tiles=args[5]//128),branch=branch,explicit_pass_configs={} if mla else {k:False for k in ['tl.ascend_auto_cv_combine','tl.ascend_auto_cv_sync','tl.ascend_auto_sync','tl.ascend_memory_planning']},target='pto',platform='A5',default_tiles=dict(block_I=64,core_num=24) if mla else dict(block_M=64,block_N=128,num_stages=14,core_num=20),prior_route_representative=prior,prior_exact_timeout=cid==prior,prior_exact_evidence=r.get('prior_exact_evidence'),status='historical_exact_timeout_retained' if cid==prior else 'individually_unverified',numerical_success=False,accepted_added=0))
(P/'EXACTKEY_SOURCE_MAPPING.json').write_text(json.dumps(rows,indent=2));print(collections.Counter(r['branch'] for r in rows))
