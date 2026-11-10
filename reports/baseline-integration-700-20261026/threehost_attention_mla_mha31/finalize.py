from pathlib import Path
import json,hashlib,datetime,collections
P=Path(__file__).resolve().parent;rows=json.loads((P/'EXACTKEY_SOURCE_MAPPING.json').read_text());at=datetime.datetime.now(datetime.timezone.utc).isoformat();runs=[]
for host,directory in [('local','local_runs'),('server1','server1_evidence/runs'),('server2','server2_evidence/runs')]:
 for g in (P/directory).glob('*/guard.json'):
  d=g.parent;j=json.loads(g.read_text());cid=j['argv'][-1];op=j['argv'][-2];log=(d/'guard.log').read_text();rc=j['returncode'];la=list(d.glob('launch_*/native.json'))
  st={75:'not_started_guard75',76:'invalidated_external_process_guard76',124:'timeout'}.get(rc,'completed' if rc==0 else 'execution_failure')
  phase='not_started' if rc==75 else 'device_synchronize' if 'ADAPTER_CALL_RETURNED' in log else 'native_call' if la else 'before_native_call'
  ck=d/'checker.json';checker=json.loads(ck.read_text()) if ck.exists() else None
  runs.append(dict(key=['tilelang-ascend',op,cid],host=host,status=st,phase=phase,guard=str(g.relative_to(P)),returncode=rc,elapsed_with_queue=j['end']-j['start'],native_receipts=[str(p.relative_to(P)) for p in la],checker=checker,inputs_saved=(d/'inputs.pt').exists(),accepted_added=0))
for r in rows:
 rr=[x for x in runs if x['key']==r['key']];r['attempts']=rr
 if rr:
  valid=[x for x in rr if x['returncode'] not in [75,76]]
  r['status']=('actual_'+valid[-1]['status']+'_'+valid[-1]['phase']) if valid else 'individually_unverified_infrastructure_blocked'
 r['numerical_failure_established']=False;r['accepted_added']=0
counts=collections.Counter(r['status'] for r in rows)
result=dict(at=at,status='scoped_execution_and_evidence_delivered',scope=31,accepted_added=0,numerical_successes=0,branch_counts=dict(collections.Counter(r['branch'] for r in rows)),status_counts=dict(counts),exactkeys=rows,attempts=runs,limits=['No sibling key inherits runtime failure or success.','Full frozen checker invoked for both new branches but did not complete; no numerical verdict.','No device PC/event state establishes exact blocked instruction.','Local guard76 run invalid; original timeout keys not rerun.'])
(P/'RESULT.json').write_text(json.dumps(result,indent=2))
delivery=dict(at=at,accepted_added=0,scope=31,findings=['Independently mapped 25 MLA and 6 MHA keys to two original builders, four control-flow branches.','Two new branch exactcases selected: H128 MLA ds-v3-single-request-float16 and causal MHA speculative-decode-float16.','Historical H64 MLA and noncausal MHA exact timeouts retained without identical replay.','Both new branches have actual native provenance; consult RESULT for final guard and checker state.','A5 header confirms hardware flag+16 routing; no missing-second-vector-set diagnosis.'],files=['RESULT.json','EXACTKEY_SOURCE_MAPPING.json','CHECKS.json','DIAGNOSTIC.md','FROZEN_HASHES.json','MHA_EXACT_RESULT.json','server1_evidence','server2_evidence','local_runs'],checks=json.loads((P/'CHECKS.json').read_text()),remaining=['No new accepted case. Parent performs registration/acceptance.','Compiler-owner diagnostic: correlate device PC/event state with identified workspace handshake waits; preserve original source until separately authorized.','Unrun siblings remain individually unverified; expansion gate not passed.'],constraints={'kernel_compiler_checker_seed_tolerance_registry_memory_modified':False,'subagents_spawned':False,'external_process_killed':False,'changed_files':'Only this task directory and matching remote private directories.'})
(P/'DELIVERY.json').write_text(json.dumps(delivery,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps({'at':at,'phase':'finished','accepted_added':0,'status_counts':dict(counts)},indent=2))
hashes={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.rglob('*') if f.is_file() and f.name!='FILEHASHES.json'};(P/'FILEHASHES.json').write_text(json.dumps(hashes,indent=2));print(json.dumps({'status_counts':dict(counts),'attempts':[(r['host'],r['key'][-1],r['returncode']) for r in runs],'hash_count':len(hashes)},indent=2))
