import pathlib,json,hashlib,ast,time
W=pathlib.Path('/root/dyq/workspace');P=pathlib.Path(__file__).resolve().parent;v=W/'ascend_baselines/tilelang-ascend'
files=['tilelang/jit/adapter/libgen.py','examples/gemm/example_gemm.py','examples/batch_gemm/batch_gemm.py','examples/topk_selector/example_topk_selector.py','examples/deepseek_v4/int8_gemm.py','examples/causal_conv1d/causal_conv1d_decode.py','examples/causal_conv1d/causal_conv1d_pto.py','examples/normalization/rms_norm.py','testing/python/language/test_tilelang_ascend_language_elementwise.py','testing/python/base/unary_op.py']
rows=[]
for rel in files:
 f=v/rel;t=f.read_text();tree=ast.parse(t);defs=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
   ds=[ast.get_source_segment(t,d) for d in n.decorator_list]
   if ds or 'reduce_runtime' in n.name:defs.append(dict(name=n.name,line=n.lineno,decorators=ds))
 rows.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),definitions=defs,arch_lines=[dict(line=i,text=l) for i,l in enumerate(t.splitlines(),1) if 'npu-arch' in l or 'cce-aicore-arch' in l]))
(P/'ORIGINAL_API_BOUNDARIES.json').write_text(json.dumps(rows,indent=2))
r=json.loads((P/'RESULT.json').read_text())
for x in r['rows']:
 if x['baseline']=='tilelang-ascend' and x['operator'] not in ['CausalConv1dFwdOp','AllFwdOp']:
  x['original_reason']=x['reason'];x['status']='blocked';x['reason']='Original/default AscendC build selects hardcoded dav-2201 on all three observed identical libgen sources. SDK header upgrade does not change this target. Forced PTO historical outcomes are not accepted. No unchanged dav-2201 device rerun.';x['current_evidence']=['ORIGINAL_API_BOUNDARIES.json','local_inventory.json','server1_inventory.json','server2_inventory.json','../tla_default_recovery23/FINDINGS.md'];x['device_rerun']=False
r['updated']=time.time();(P/'RESULT.json').write_text(json.dumps(r,indent=2));print(len(rows))
