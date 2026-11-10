import pathlib,json,subprocess,os,sys
D=pathlib.Path(__file__).resolve().parent;W=D.parents[2]
s=json.loads((D/'SCOPE.json').read_text()); policy=json.loads((W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/user_amnesty_0323/POLICY.json').read_text());ledger=json.loads((W/'.cache/all-pairs/full_rollout_20261024/local24/group1/resume25_common_repair/continuation0320/TARGET_SETS.json').read_text())
for x in s:
 op,cid=x['operator'],x['case_id'];key=x['key']
 if len(sys.argv)>1 and op!=sys.argv[1]:continue
 assert key not in policy['amnesty_keys']
 for field in ['original_verified_target','proven_excluded_non_target']:
  assert not any((r.get('key') if isinstance(r,dict) else r)==key for r in ledger.get(field,[]))
 if len(sys.argv)>2 and cid!=sys.argv[2]:continue
 d=D/(op+'--'+cid+(('--'+sys.argv[3]) if len(sys.argv)>3 else ''))
 if d.exists():continue
 d.mkdir();cmd=['npu-guard','run','--owner','selected-integration-20261026/nn_norm','--wait','3600','--timeout','300','--',str(W/'.tools/baselines/baseline-run'),'ops-nn','python',str(D/'packet/run_case.py'),op,cid,str(d/'output')]
 (d/'COMMAND.json').write_text(json.dumps(cmd));env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',FLA_NPU_DISABLE_PTH='1',BASELINE_PROGRESS_FILE=str(d/'progress.json'))
 with (d/'guard.log').open('w') as f:r=subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
 (d/'GUARD_RESULT.json').write_text(json.dumps({'returncode':r.returncode}));print(key,r.returncode,flush=True)
 if r.returncode:break
