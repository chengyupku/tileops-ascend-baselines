import unittest
import json
import os
import tempfile
import time
from pathlib import Path
from run_baseline_cache import (classify_failure, quarantine_key, identity, live_external_blockers,
                                completed_ids, record_deployment_failure, assigned_history, record_timeout)
from baseline_cache_guard_binding import bind_guard, validate_binding
from baseline_cache_env import selected_environment


class CacheControlTest(unittest.TestCase):
    def test_timeout_holds_only_failed_pair_and_does_not_repeat_completed_work(self):
        bad = {'id':'bad', 'key':['lib','Op','slow'], 'status':'failed'}
        skipped = {'id':'skip', 'key':['lib','Op','other'], 'status':'temporarily_skipped_library'}
        q = {}
        record_timeout(q, bad, '/failed')
        self.assertEqual(quarantine_key(bad, q), 'pair:bad')
        self.assertIsNone(quarantine_key(skipped, q))
        self.assertIsNone(quarantine_key({'id':'other','key':['lib','Other','x']}, q))
        self.assertEqual(completed_ids({'bad':bad,'skip':skipped}, {}, q, {'cached'}), {'bad','cached'})

    def test_ops_math_oracle_does_not_load_expanded_graph_opp(self):
        graph = '/root/dyq/workspace/.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
        env, evidence = selected_environment({'key':['ops-math','FFTC2CFwdOp','case']},
            {'ASCEND_CUSTOM_OPP_PATH':graph, 'LD_LIBRARY_PATH':graph+'/op_api/lib:/keep/lib'})
        self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'], '')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/keep/lib')

    def test_repartition_preserves_history_but_counts_only_current_owner(self):
        rows = [{'id':'stay','status':'cached'}, {'id':'move','status':'failed'}]
        cache = [{'task_id':'stay'}, {'task_id':'other'}]
        latest, cached = assigned_history([{'id':'stay'}], rows, cache)
        self.assertEqual(set(latest), {'stay'})
        self.assertEqual(cached, {'stay'})
        self.assertEqual(len(rows), 2)

    def test_exact_deployment_hold_does_not_block_other_cases(self):
        q = {'pair:bad': {'reason':'no accepted route for this exact case'}}
        self.assertEqual(quarantine_key({'id':'bad','key':['lib','Op','a']}, q), 'pair:bad')
        self.assertIsNone(quarantine_key({'id':'good','key':['lib','Op','b']}, q))

    def test_contamination_is_not_numerical_failure(self):
        m = {'status':'correctness_not_verified'}
        self.assertEqual(classify_failure(76,m,{'conflict_detected':True}), 'performance_invalid_external_conflict')
    def test_wait_does_not_quarantine_a_baseline(self):
        self.assertEqual(classify_failure(75,{},{}), 'resource_wait')
        self.assertEqual(classify_failure(1,{'status':'correctness_not_verified'},{}), 'correctness_not_verified')
    def test_scoped_quarantine_leaves_other_ops_eligible(self):
        q={'catlass::AddFwdOp': {'reason':'no registered Add provider'}}
        self.assertEqual(quarantine_key({'key':['catlass','AddFwdOp','x']},q),'catlass::AddFwdOp')
        self.assertIsNone(quarantine_key({'key':['catlass','GemmFwdOp','y']},q))
        self.assertIsNone(quarantine_key({'key':['vllm-ascend','AddFwdOp','x']},q))

    def test_missing_native_extension_does_not_suppress_triton_providers(self):
        q, failures = {}, {}
        for n in range(3):
            record_deployment_failure(q, failures,
                {'id':str(n), 'key':['vllm-ascend','FusedTopKFwdOp',str(n)]}, '/failed/'+str(n))
        self.assertIn('vllm-ascend::FusedTopKFwdOp',q)
        self.assertIsNone(quarantine_key({'key':['vllm-ascend','RMSNormFwdOp','x']},q))
        self.assertNotIn('vllm-ascend',q)

    def test_distinct_operators_or_repeated_pair_do_not_accumulate(self):
        q, failures = {}, {}
        for op in ('TopK','RMSNorm','Add'):
            task={'id':op,'key':['lib',op,'case']}
            for _ in range(3):
                record_deployment_failure(q,failures,task,'/failed')
        self.assertEqual(q,{})

    def test_released_skips_resume_without_repeating_any_cached_pair(self):
        latest={
            'skip':{'key':['lib','Good','a'],'status':'temporarily_skipped_library'},
            'hold':{'key':['lib','Bad','b'],'status':'temporarily_skipped_library'},
            'failed':{'key':['lib','Good','c'],'status':'correctness_not_verified'},
            'cached':{'key':['lib','Good','d'],'status':'cached'}}
        done=completed_ids(latest,{}, {'lib::Bad':{}}, {'cached','cache_fsynced_before_crash'})
        self.assertEqual(done,{'hold','failed','cached','cache_fsynced_before_crash'})

    def test_existing_library_hold_is_preserved(self):
        q={'lib':{'reason':'independently confirmed device hang'}}
        failures={}
        for n in range(3):
            record_deployment_failure(q,failures,{'id':str(n),'key':['lib','Op',str(n)]},'/failed')
        self.assertEqual(q['lib']['reason'],'independently confirmed device hang')
    def test_external_gate_checks_native_identity_not_just_pid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'blockers.json'
            native = identity(os.getpid())
            p.write_text(json.dumps({'processes':[native]}))
            self.assertEqual(live_external_blockers(p),[native])
            native['start_ticks'] += 1
            p.write_text(json.dumps({'processes':[native]}))
            self.assertEqual(live_external_blockers(p),[])
    def test_recorded_batch_rule_does_not_match_another_user(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blockers.json'
            p.write_text(json.dumps({'batch_rules':[{'uid':999999,'program_argument':'unittest','evidence':'test'}]}))
            self.assertEqual(live_external_blockers(p),[])

    def test_guard_binding_detects_receipt_and_implementation_changes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            impl, receipt = Path(d)/'guard.py', Path(d)/'receipt.json'
            impl.write_text('guard v1')
            receipt.write_text(json.dumps({'phase':'finished','performance_valid':True,
                'workload_exit_code':0,'owner':'campaign/local/pair'}))
            row={'cache_key':'cache','task_id':'pair','guard_receipt':str(receipt),'started_at':time.time()}
            fingerprint={'files':{str(impl):hashlib.sha256(impl.read_bytes()).hexdigest()},'stamps':{str(impl):[1,2,8,0,0]},'observed_at':time.time()}
            binding=bind_guard(row,fingerprint,fingerprint)
            self.assertEqual(validate_binding(row,binding),[])
            impl.write_text('guard v2')
            self.assertIn('guard implementation changed',str(validate_binding(row,binding)))
            receipt.write_text('{}')
            self.assertIn('guard receipt changed',str(validate_binding(row,binding)))

    def test_retrospective_binding_cannot_claim_new_guard_was_used(self):
        row={'started_at':1}
        fingerprint={'files':{'guard':'hash'},'stamps':{'guard':[1,2,3,2_000_000_000,2_000_000_000]}}
        with self.assertRaisesRegex(ValueError,'newer guard'):
            bind_guard(row,fingerprint,fingerprint,retrospective=True)

    def test_reviewed_nn_vendor_is_selected_without_duplicate_opp(self):
        task={'key':['ops-nn','RMSNormFwdOp','llama-8b-decode-bfloat16']}
        env,_=selected_environment(task,{'ASCEND_CUSTOM_OPP_PATH':'incorrect:duplicate'})
        self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'],env['OPS_NN_VENDOR'])
        self.assertTrue(env['OPS_NN_VENDOR'].endswith('/vendors/classicpilot_nn'))
        self.assertNotIn(':',env['ASCEND_CUSTOM_OPP_PATH'])

    def test_fla_fp16_small_bmm_selects_native_opp_before_import(self):
        task={'key':['flash-linear-attention-npu','BmmFwdOp','square-128-float16'],
              'case':{'tensors':{'a':[[8,128,128],'float16']},'params':{}}}
        env,evidence=selected_environment(task,{})
        self.assertEqual(evidence['mode'],'gdn')
        vendor=Path(env['ASCEND_CLASSIC_FLA_PACKAGE'])/'opp/vendors/fla_npu_transformer'
        config=json.loads((vendor/'op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json').read_text())
        self.assertIn('ChunkFwdO',config)
        self.assertEqual(env['FLA_NPU_DISABLE_PTH'],'1')


if __name__ == '__main__':
    unittest.main()
