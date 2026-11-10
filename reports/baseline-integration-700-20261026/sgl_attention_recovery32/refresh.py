from pathlib import Path
import json,datetime,collections
P=Path(__file__).resolve().parent;rows=json.loads((P/'SCOPE.json').read_text());key=lambda r:(r['baseline'],r['operator'],r['case_id']);by={key(r):r for r in rows}
for r in rows:r.update(status='blocked',registered=False,checker_receipt=None,guard_receipt=None,timings={'queue_seconds':None,'device_seconds':None,'compile_seconds':None})
for a in sorted((P/'attempts').glob('*/GUARD_RESULT.json'),key=lambda f:json.loads(f.read_text())['started_epoch_local']):
 g=json.loads(a.read_text());r=by[tuple(g['key'])];c=P/'runs'/a.parent.name/'checker.json';r['guard_receipt']=str(a.relative_to(P));r['checker_receipt']=str(c.relative_to(P)) if c.exists() else None
 if not c.exists():r.update(status='blocked',reason='Guard or runner did not produce checker receipt; see preserved attempt.');continue
 d=json.loads(c.read_text());v=d.get('verification',{});r.update(api=d.get('api',r.get('api')),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',r.get('total_outputs')),factory=json.loads((c.parent/'ROUTE.json').read_text()).get('factory'))
 if (g['exit_code']==0 or (a.parent.name.startswith('reps03-') and g['exit_code']==1 and d.get('status')=='verified')) and d['status']=='verified' and v.get('checked_outputs')==v.get('total_outputs') and v.get('checked_outputs',0)>0 and d.get('inputs_preserved') and not d.get('oracle_source_changed') and not d.get('provider_source_changed'):
  r.update(status='verified_unregistered',reason='Full canonical checker verified all outputs/state with authorized one-line diagnostic-only CPU override; parent registration pending.')
 else:r.update(status='failed' if g['exit_code'] not in (75,76,78) else 'blocked',reason=f"guard={g['exit_code']}; checker={d.get('status')}; stage={d.get('stage')}; error={d.get('error')}")
for r in rows:
 if r.get('checker_receipt') and json.loads((P/r['checker_receipt']).read_text()).get('status')=='source_changed':
  r['status']='blocked';r['reason']='Full numerical comparison passed, but original checker reports provider_source_changed during external attention14/GDN30 registrations. Guard1 retained; not accepted and not numerically rerun. See NUMERICAL_PASS_SOURCE_RACE.json.'
for r in rows:
 if r.get('checker_receipt'):
  c=P/r['checker_receipt'];d=json.loads(c.read_text());r['timings']['checker_wall_seconds']=d.get('wall_seconds')
  m=c.parent/'GUARD_RUNNING.json'
  if m.exists():
   g=json.loads(m.read_text())
   r['timings']['queue_seconds']=(datetime.datetime.fromisoformat(g['acquired_at'])-datetime.datetime.fromisoformat(g['requested_at'])).total_seconds()
for r in rows:
 if r['operator']=='DSADecodeWithKVCacheFwdOp' and r['checker_receipt'] is None:
  r['reason']='Canonical DSA recovery not yet attempted; sequential guarded expansion pending after successful original DSA representatives. No numerical failure inferred.'
for i in [0,1,4]:
 rows[i]['reason']='Original prefix-valid store requires matching storage dtype and has no FP8 cache quantization divisor; original decode/prefill APIs have no supplied FP8 cache dequantization scale contract. This cannot be repaired by lossless metadata/layout binding.'+(' Required logits softcap is also absent from the original attention API.' if i==4 else '')
rows[3]['reason']='Original attention_sinks_prefill_triton/decode_gqa APIs do not expose the required logits softcap; adding softcap arithmetic would be a new mathematical path.'
for i in [30,31]:rows[i]['reason']='Original A5/Ascend950 build disables SGL_KERNEL_ENABLE_A3_ONLY_OPS; unidex host, kernel and registration are A3-only. Original server2 lib c8026d31 has no unidex symbol/registration (GATHER_CMAKE.txt, GATHER_A5_EXCLUSION.txt, GATHER_ELF.txt). Enabling A3 compiler gates or replacing kernel/bridge is outside allowed binding-only changes.'
for r in rows:
 r['key']={k:r[k] for k in ('baseline','operator','case_id')}
 if r['checker_receipt']:
  r['evidence']=[r['checker_receipt'],r['guard_receipt'],'DIAGNOSTIC_ONLY.patch','SOURCE_HASHES.json','CONTRACT_MATCH.json']
 else:
  r['evidence']=[str((P.parent/'sgl_attention32'/e).resolve()) if (P.parent/'sgl_attention32'/e).exists() else e for e in r.get('evidence',[])]
rows[8]['reason']='Original prefix-valid store only copies matching storage dtypes and exposes no FP8 quantization divisor. Native RoPE plus store is permitted, but cannot satisfy this FP8 cache scale contract without new arithmetic.'
counts=dict(collections.Counter(r['status'] for r in rows));(P/'RESULT.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'rows':rows},indent=2));print(counts)
