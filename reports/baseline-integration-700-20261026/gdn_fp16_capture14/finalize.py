from pathlib import Path
import json,datetime,hashlib,ast
W=Path(__file__).resolve().parent;now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[]
old=[json.loads(x) for x in (W/'RESULT.jsonl').read_text().splitlines()]
for scope in json.loads((W/'SCOPE.json').read_text()):
 case=scope['case_id'];attempt={'decode-b1-float16':'attempt01_decode','varlen-ragged-grouped-float16':'attempt02_varlen'}.get(case)
 row=dict(baseline='ops-transformer',operator='GDNFwdOp',case_id=case,status='not_run_family_blocked',registered=False,accepted=False,total_outputs=2,outputs_diagnosed=0,outputs_passed=0,api_queue_seconds=None,guard_receipt=None,checker_receipt=None,reason='No legal fix established after authorized probes; other scoped cases were not executed.')
 if attempt:
  a=W/attempt;d=json.loads((a/'CPU_ANALYSIS.json').read_text());c=json.loads((a/'checker.json').read_text())
  row.update(status='diagnostic_failed' if case.startswith('decode') else 'metadata_probe_failed',attempt=attempt,outputs_diagnosed=2,outputs_passed=sum(x['passed'] for x in d['outputs']),input_seed=c['input_seed'],guard_receipt=attempt+'/GUARD_RESULT.json',checker_receipt=attempt+'/checker.json',cpu_analysis=attempt+'/CPU_ANALYSIS.json',reason='Original canonical checker failed. Independent per-output CPU diagnostics retain both actual outputs; no acceptance or registration.')
 rows.append(row)
(W/'RESULT.json').write_text(json.dumps(rows,indent=2)+'\n')
with (W/'RESULT.jsonl').open('a') as f:
 for row in rows:
  if row['case_id'] not in [x['case_id'] for x in old]:f.write(json.dumps(row)+'\n')
for p in W.glob('*.py'):ast.parse(p.read_text())
for p in (W/'package').glob('*.py'):ast.parse(p.read_text())
checks=dict(at=now,scope_rows=len(rows),unique_exact_keys=len({(r['baseline'],r['operator'],r['case_id']) for r in rows}),device_attempts=2,accepted_added=0,registered_added=0,bf16_runs=0,python_ast='passed',all_frozen_files_unchanged=all(r['frozen_files_unchanged'] for r in json.loads((W/'POST_AUDIT_ALL.json').read_text())),actual_binary_prefrozen=all(b['hash_was_prefrozen'] for r in json.loads((W/'POST_AUDIT_ALL.json').read_text()) for b in r['actual_dispatched_binaries']),shared_files_modified=False,api_queue_seconds=None)
(W/'FINAL_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now,phase='completed',attempts=2,accepted_added=0)))
