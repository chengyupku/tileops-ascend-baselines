import json,pathlib,datetime,collections
P=pathlib.Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text()); rows=[]; probes={}
for d in sorted((P/'runs').iterdir()):
 if not (d/'guard.json').exists():continue
 g=json.loads((d/'guard.json').read_text());meta=json.loads((d/'GUARD_RUNNING.json').read_text()) if (d/'GUARD_RUNNING.json').exists() else {}
 t=lambda n:datetime.datetime.fromisoformat(meta[n]).timestamp() if n in meta else None
 req,acq,run=t('requested_at'),t('acquired_at'),t('updated_at')
 timings=dict(queue_seconds=acq-req if req and acq else None,physical_idle_wait_seconds=run-acq if run and acq else None,held_seconds=g['finished']-acq if acq else None,device_process_seconds=g['finished']-run if run else None,compile_seconds=None,device_seconds=None,wall_seconds=g['finished']-g['started'])
 probes[d.name]=dict(returncode=g['returncode'],timings=timings,key=g['key'])
for r in scope:
 op,cid=r['operator'],r['case_id']
 if op=='MLADecodeWithKVCacheFwdOp':
  probe='mla-tp-b1-float16';api='examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py:sparse_attention_fwd'
 elif cid.startswith('gemma'):
  probe='gemma-2b-8k-bs1-float16';api='examples/flash_attention/flash_attn_bhsd.py:flash_attention_fwd'
 elif op in ['GQADenseFwdOp','GQAPagedFwdOp','GQAVarlenFwdOp'] and not cid.startswith('gpt-oss') and any(k.startswith('qwen3-30b-1k-float16-') for k in probes):
  probe='qwen3-30b-1k-float16';api='examples/gqa_fwd_varlen/gqa_fwd_varlen.py:flashattn'
 else:
  probe='serving-kv512-float16';api='examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py:gqa_sink_fwd'
 matches=[k for k in probes if k.startswith(probe+'-')];last=matches[-1] if matches else None
 d=P/'runs'/last if last else None;c=json.loads((d/'checker.json').read_text()) if d and (d/'checker.json').exists() else None
 code=probes[last]['returncode'] if last else None
 if code==124:reason='Original full API compiled with public PTO backend binding, but complete canonical checker did not finish within the 300-second foreground guard. No numerical success. Shared route stopped; sibling shapes not replayed.'
 elif c:reason=c.get('error','No complete numerical pass')
 else:reason='Awaiting representative route completion; no numerical success claimed.'
 rows.append(dict(baseline=r['baseline'],operator=op,case_id=cid,status='failed' if cid==probe and last else 'blocked',reason=reason,api=api,guard_receipt=str((d/'guard.json').relative_to(P)) if d else None,checker_receipt=str((d/'checker.json').relative_to(P)) if c else None,checked_outputs=0,total_outputs=1,registered=False,factory=None,changed_files=['adapter.py','run_case.py','launch.py'],timings=probes[last]['timings'] if last and cid==probe else dict(queue_seconds=None,device_seconds=None,compile_seconds=None),representative_evidence=last,source_evidence=r['evidence']))
obj=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='representative_route_validation',rows=rows,probes=probes,counts=dict(collections.Counter(r['status'] for r in rows)))
f=P/'RESULT.json.tmp';f.write_text(json.dumps(obj,indent=2,ensure_ascii=False));f.replace(P/'RESULT.json');print(obj['counts'])
