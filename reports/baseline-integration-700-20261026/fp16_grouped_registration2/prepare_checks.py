from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parent/'fp16_bmm_registration1'
s=(B/'cpu_checks.py').read_text().replace('integration700_fp16_bmm_registration1','integration700_fp16_grouped_registration2').replace('FP16_BMM','FP16_GROUPED')
s=s.replace("for op in ('BmmFwdOp',):","for op in ('GroupedGemmFwdOp',):").replace('assert len(cs)==1','assert len(cs)==2').replace('assert len(previous)==1','assert len(previous)==2').replace("c=next(c for c in cs if c._call.signature.name==k[1])","c=next(c for c in cs if c.id==k[2] or 'old-'+c.id==k[2])")
s=s.replace("('a','b','d')","('a','b','batch_sizes','batch_offsets','output')")
s=s.replace("and not frozen._call.present('out')","and not frozen._call.present('out')")
s=s.replace("dict(cs[0]._call.indices,B=1)","dict(cs[0]._call.indices,G=1)")
a=s.index('# Execute exact original native nested function AST');b=s.index("# Admission's original code",a)
s=s[:a]+(O/'grouped_checks.txt').read_text()+'\n'+s[b:]
s=s.replace("admission['input_elements']==67108864 and admission['output_elements']==33554432","admission['input_elements']==285212672 and admission['output_elements']==16777216")
s=s.replace("fp16_local_remaining3/square-2k-float16/result/RUNTIME.json","fp16_grouped_budget900_2/moe-dgrad-float16/result/RUNTIME.json")
s=s.replace('all one','all two').replace('one real manifest','two real manifest').replace('1 real canonical','2 real canonical').replace('1 previous-route','2 previous-route').replace('batch/rawparams','group/rawparams').replace('full immutable batch envelope','full immutable grouped envelope')
# Missing or incorrect owner in real guard group must reject during construction.
needle="lock.write_text(json.dumps(meta));assert M._guard_identity() is not None"
s=s.replace(needle,needle+"\n  for owner in (None,'wrong'):\n   if owner is None:os.environ.pop('FP16_GROUPED_GUARD_OWNER',None)\n   else:os.environ['FP16_GROUPED_GUARD_OWNER']=owner\n   assert M._in_guard_group() and M._guard_identity() is None\n   reject(lambda:M.make(cs[0]))\n  os.environ['FP16_GROUPED_GUARD_OWNER']='cpu-fixture-only'")
s=s.replace('# Full real pin check is CPU file hashing, not a mocked runtime validation.',(O/'additional_checks.txt').read_text()+'\n# Full real pin check is CPU file hashing, not a mocked runtime validation.')
(O/'cpu_checks.py').write_text(s)
