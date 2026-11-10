from pathlib import Path
import json,hashlib,datetime,ast
P=Path(__file__).resolve().parent;W=P.parents[2];O=P/'runs/probe01--GDNFwdOp--prefill-narrow-state-bfloat16/output'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((O/'checker.json').read_text());a=json.loads((P/'CAPTURE_ANALYSIS.json').read_text());g=json.loads((P/'GUARD_RESULT.json').read_text());pre=json.loads((P/'PREFLIGHT_LOCAL.json').read_text())
local={k:h(W/k)==v for k,v in pre['original_hashes'].items()};assert all(local.values())
assert g['exit_code']==0 and r['status']=='private_checker_diagnostic_pass' and a['all_inputs_bitwise_identical']
assert r['verification']['checked_outputs']==r['verification']['total_outputs']==2
assert not r['oracle_source_changed'] and not r['provider_source_changed']
assert h(P/'integration700_sgl_gdn_remote38.py')==h(P.parent/'sgl_gdn_register30/frozen_verified/integration700_sgl_gdn_remote38.py')
for f in P.glob('*.py'):ast.parse(f.read_text())
(P/'FINAL_CHECKS.json').write_text(json.dumps(dict(local_shared_sources_unchanged=local,remote_shared_sources=json.loads((P/'SHARED_UNCHANGED.json').read_text()),frozen_adapter_byte_identical=True,python_ast_parse=True,guard_true_exit_code=0,original_math_and_validator_unchanged=True,private_checker_sha256=h(P/'checker.private.py'),original_checker_sha256=h(P/'checker.original.py'),capture_analysis_exit_code=0,cpu_analysis_note='Initial system python lacked torch (exit 1); same post-analysis script completed under .venv CPU torch, device imports blocked; no NPU retry',feedback_files=[str(p) for p in P.glob('*FEEDBACK*')]),indent=2))
entries={p.stem:json.loads(p.read_text()) for p in (O/'capture').glob('*.json')}
timing={k:v['time'] for k,v in entries.items()}
delivery=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='private_checker_diagnostic_pass',scope_count=6,scope_meaning='Six potential affected keys only; no execution ownership or acceptance of remaining scope',actual_probes=1,remaining_unrun=5,accepted_added=0,registered_added=0,shared_changes_applied=False,key=['sgl-kernel-npu','GDNFwdOp','prefill-narrow-state-bfloat16'],input_seed=r['input_seed'],input_origin='fresh canonical NPU generation in this single guarded process; no historical payload or synthetic inputs',api=r['api'],guard_exit_code=g['exit_code'],checker_verification=r['verification'],input_preservation=a['all_inputs_bitwise_identical'],outputs=a['outputs'],source_hashes=pre['original_hashes'],original_checker_sha256=h(P/'checker.original.py'),private_checker_sha256=h(P/'checker.private.py'),adapter_sha256=h(P/'integration700_sgl_gdn_remote38.py'),timings=dict(checker_wall_seconds=r['wall_seconds'],guard_ssh_elapsed_seconds=g['end_epoch']-g['start_epoch'],reference_cpu_with_capture_seconds=timing['reference_cpu_inputs_after']-timing['reference_cpu_inputs_before'],candidate_with_capture_seconds=timing['candidate_inputs_after']-timing['candidate_inputs_before']),timing_limits='Diagnostic capture/profile overhead; no benchmark. Host wall clocks differ (~93 s), never subtract cross-host timestamps.',binary_identity='BINARY_IDENTITIES.json: 164 mapped runtime/library files hashed after process exit; no custom fused SGL kernel claim',evidence=['checker.private.diff','FINAL_CHECKS.json','CAPTURE_ANALYSIS.json','GUARD_RESULT.json','SHARED_HASHES_before.json','SHARED_HASHES_after.json','BINARY_IDENTITIES.json',str(O.relative_to(P)/'checker.json'),str(O.relative_to(P)/'native_calls.jsonl'),str(O.relative_to(P)/'native.jsonl'),str(O.relative_to(P)/'capture')],remaining_issues=['Shared checker CPU-dispatch authorization/review remains separate; original checker was not passed.', 'Other five scope keys and huge q35 cases untested; no extrapolated acceptance.', 'CPU backend floating-point rounding may differ; this real canonical case passed unchanged full validator.', 'One private diagnostic only, no performance/profiling or throughput claim.'])
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2))
(P/'DELIVERY.md').write_text('''# Private GDN canonical diagnostic

Status: **private_checker_diagnostic_pass**. Exactly one guarded probe, true exit 0. Added acceptance/registration: **0**. Six keys are only a potential impact scope; other five were not run.

The original `prefill-narrow-state-bfloat16` case used canonical NPU-generated inputs (seed 2322693802), B=1, S=4096, H=HV=16, K=V=64, BF16; optional initial state remains absent. The frozen full `chunk_gated_delta_rule_native` API executed on NPU with original dtype. No later l2 adapter, kernel, schedule, tile, target, seed, tolerance or mathematical changes.

`gdn_cpu_dispatch.patch` applied cleanly to original checker hash `3b9dd9dde43e5ff6c101496b714bd9d2710a6758264d4f851dd74da829e4bb9f`. The private checker additionally fixes its relocated source root, captures diagnostic copies, and labels success diagnostic-only. `checker.private.diff` is the complete reviewable diff. GDN reference and numerics source hashes are unchanged. Original FP64 reference and complete validator execute on CPU; original candidate receives original NPU arguments/reset.

All six before/after input snapshots (reference NPU, reference CPU, candidate NPU) have identical tensor content hashes. Full o and final_state payloads are retained as .pt files with per-tensor and file SHA256. Both outputs were checked by original validation: o max absolute error 7.62939453125e-6, normalized squared error 1.0666695353174115e-10; FP32 state max absolute error 1.862645149230957e-9, normalized squared error 1.6613341304016735e-15. Original atol/rtol 0.016 and normalized bound 0.001 are unchanged.

Evidence includes real guard metadata, true exit, full checker receipt, original native API call/return trace, 2225 CANN attribution events, loaded source hashes and 164 mapped binary hashes. 36 monitored shared files were unchanged before/after; local proposal source pins also remain unchanged. CPU preflight used stdlib only. CPU post-analysis blocked torch_npu/ACL imports and ran after guard release. No NPU replay occurred.

Reference CPU stage including capture took about 0.51 seconds; complete checker took 32.87 seconds and guarded SSH call 45.01 seconds. These are diagnostic costs, not kernel timings. Local/server2 wall clocks differ, so cross-host timestamps must not be subtracted.

All changes/artifacts are under the assigned local/remote gdn_oracle_probe6 directories. Main files: checker.private.py and diff, capture.py, frozen adapter copy, run_case.py, setup/transport/analysis scripts, DELIVERY.json, FINAL_CHECKS.json, CAPTURE_ANALYSIS.json, source/binary manifests, and runs/probe01--GDNFwdOp--prefill-narrow-state-bfloat16/output/ captures and receipts. Original logs/cache remain in the remote assigned directory; compact native evidence and payloads were copied locally.

Remaining: shared application still needs separate review; this is not an original-checker pass or formal acceptance. No result is asserted for the other five keys or large q35 shapes. No benchmark claim.
''')
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=delivery['utc'],stage='complete; private diagnostic only',feedback_read='no FEEDBACK file present',accepted_added=0)))
manifest={str(p.relative_to(P)):h(p) for p in P.rglob('*') if p.is_file() and p.name!='ARTIFACT_HASHES.json'}
(P/'ARTIFACT_HASHES.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(status=delivery['status'],artifacts=len(manifest),accepted_added=0)))
