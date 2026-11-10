import pathlib,json,hashlib,ast,difflib,datetime,re
P=pathlib.Path(__file__).resolve().parent;R=P.with_name('sgl_attention_recovery32');W=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
C=load(R/'runs/dsa07_11/checker.json');key=[C['library'],C['op'],C['case_id']];init='TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'
assert sha(R/'runs/dsa07_11/checker.json')==load(R/'NUMERICAL_PASS_SOURCE_RACE.json')['checker_sha256']
before=(R/'provider_init.before.py').read_bytes();after=(R/'provider_init.after_race.py').read_bytes()
assert hashlib.sha256(before).hexdigest()==C['provider_source_sha256'][init]
assert hashlib.sha256(after).hexdigest()==C['provider_source_sha256_after'][init]
assert after.startswith(before)
added=after[len(before):].decode(); keys={}
for n in ast.parse(added).body:
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Dict):
  keys[n.targets[0].id]=ast.literal_eval(n.value)
assert C['case_id'] not in keys['_ar14_keys']['DSADecodeWithKVCacheFwdOp']
assert 'DSADecodeWithKVCacheFwdOp' not in keys['_integration700_sgl_gdn_register30_keys']
changes={k:{'before':C['provider_source_sha256'].get(k),'after':C['provider_source_sha256_after'].get(k)} for k in C['provider_source_sha256'].keys()|C['provider_source_sha256_after'].keys() if C['provider_source_sha256'].get(k)!=C['provider_source_sha256_after'].get(k)}
assert len(changes)==2
assert C['inputs_preserved'] and not C['oracle_source_changed'] and C['verification']['checked_outputs']==C['verification']['total_outputs']==1
assert not any(C.get(k) for k in ['error','sync_error','cleanup_error','traceback','unchecked_reason'])
assert C['input_seed']==int.from_bytes(hashlib.sha256(f"{C['op']}::{C['case_id']}".encode()).digest()[:4],'big')
assert sha(R/'dsa_adapter.py')==C['source_sha256']['reports/baseline-integration-700-20261026/sgl_attention_recovery32/dsa_adapter.py']
original=(R/'numerics.original.py').read_text();private=(R/'numerics_private.py').read_text()
assert private==original.replace('a, b = a.to(dtype), b.to(dtype)','a, b = a.to(device="cpu", dtype=dtype), b.to(device="cpu", dtype=dtype)')
assert hashlib.sha256(original.encode()).hexdigest()==C['diagnostic_only_override']['original_sha256']
assert hashlib.sha256(private.encode()).hexdigest()==C['diagnostic_only_override']['private_sha256']
remote={x['path']:x for x in load(P/'REMOTE_READ.json')};RW='/opt/tileops-all-pairs/workspace/'
for k,h in C['oracle_source_sha256'].items():assert remote[RW+k]['sha256']==h,(k,h)
for k in [init.replace('__init__.py','integration700_sgl_gdn_register30.py')]:assert remote[RW+k]['sha256']==C['provider_source_sha256_after'][k]
assert remote[RW+'reports/baseline-integration-700-20261026/sgl_attention_recovery32/runs/dsa07_11/checker.json']['sha256']==sha(R/'runs/dsa07_11/checker.json')
artifacts=load(R/'runs/dsa07_11/NATIVE_ARTIFACTS.json')
for a in artifacts:assert sha(R/a['saved_path'])==a['sha256']
events=[json.loads(l) for l in (R/'runs/dsa07_11/native.jsonl').open()]; native=[x for x in events if '_paged_mla_fwd_kernel' in x['text']]
assert len(native)==128
launch=[x for x in native if 'StreamLaunchKernelV1:' in x['text']];assert len(launch)==64
for x in launch:
 stamp=re.search(r'(2026-10-26)-([0-9:]+)\.(\d+)\.(\d+)',x['text'])
 dt=datetime.datetime.fromisoformat(stamp[1]+'T'+stamp[2]+'.'+stamp[3]+stamp[4]+'+00:00')
 assert datetime.datetime.fromisoformat(C['started_at']) <= dt <= datetime.datetime.fromisoformat(C['finished_at'])

loaded=load(R/'runs/dsa07_11/LOADED_MODULES.json')['modules'];dep=loaded['sgl_kernel_npu.attention.decode_attention'];assert dep['sha256']==C['source_sha256']['reports/baseline-integration-700-20261026/sgl_attention_recovery32/libraries/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py']
assert sha(R/'source_evidence/python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py')==dep['sha256']
for x in [dep,loaded['sgl_kernel_npu']]:assert remote[x['path']]['sha256']==x['sha256']
runner=(R/'runs/dsa07_11/RUNNER.py').read_text();assert 'torch.npu.synchronize()' in runner and "(work/'LOADED_MODULES.json').write_text" in runner
report={'key':key,'source_reconciled':True,'historical_before_sha256':hashlib.sha256(before).hexdigest(),'historical_after_sha256':hashlib.sha256(after).hexdigest(),'changes':changes,'added_exact_keys':keys,'target_outside_changed_routes':True,'run_binding':'RUNNER.py imports dsa_adapter then route_install.install assigns sgl.PROVIDERS[op]=factory in this process; filesystem registrations do not reload it.','guard_returncode':1,'child_returncode':1,'child_returncode_basis':'checker main sets rc=1 for source_changed; guard propagated 1; guard.log final status source_changed and no execution/sync/cleanup exception','verification':C['verification'],'canonical_seed':C['input_seed'],'inputs_preserved':True,'oracle_unchanged':True,'sync_cleanup':'checker finally synchronize and optional cleanup have no errors; runner finally synchronized before recorded LOADED_MODULES. Adapter has no cleanup/reset or persistent state.','diagnostic_only':'one max_abs CPU diagnostic conversion; validators byte-identical','native_launch_records':len(launch),'native_prepare_records':len(native)-len(launch),'native_sample':native[:2]+native[-2:],'loaded_modules':loaded,'native_artifacts':artifacts,'native_attribution':'64 calls for 4096 queries batched by 64; original loaded decode_mla launches _paged_mla_fwd_kernel; isolated per-run cache holds matching named npubin+metadata+ttir. Launch logs do not expose a binary content hash. Same attribution standard as accepted register10.','math_state_contract':'Exact key has dim_tail=0,stride_kv=1,Hkv=1,is_causal=True. Sorted unique valid causal selection matches reference scatter set mask; packed gathered KV values are unchanged, zero rotary tails add zero to dot product, true lengths exclude padding, scale remains 512**-0.5. Full original eight-argument decode_mla writes each independent query output; no state mutation of q/kv/indices, no math fallback, no kernel/compiler/target edits.','parent_accepted_added':0,'npu_runs':0}
(P/'SOURCE_RECONCILIATION.json').write_text(json.dumps(report,indent=2))
for n in ['provider_init.before.py','provider_init.after_race.py','dsa_adapter.py']:(P/n).write_bytes((R/n).read_bytes())
(P/'SOURCE_RACE.diff').write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='historical-before',tofile='historical-after')))
(P/'checker.original.json').write_bytes((R/'runs/dsa07_11/checker.json').read_bytes())
print('AUDIT PASS',len(launch),'native launches')
