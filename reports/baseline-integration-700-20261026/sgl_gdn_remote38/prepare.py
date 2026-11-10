from pathlib import Path
import shutil
P=Path(__file__).resolve().parent; old=P.parent/'sgl_gdn_recovery03'; lin=P.parent/'sgl_linalg36'
for f in ['numerics_private.py','numerics.original.py','DIAGNOSTIC_ONLY.patch','CPU_DIAGNOSTIC_PREFLIGHT.json','native_log_capture.py','route_install.py','CONTRACTS.json']:
 shutil.copyfile(old/f,P/f)
for f in ['baseline-run','run_batch.py','guard_batch.py']:
 (P/f).write_text((lin/f).read_text().replace('sgl_linalg36','sgl_gdn_remote38'))
s=(old/'integration700_sgl_gdn_recovery03.py').read_text().replace("ROOT=LIBRARIES/'sgl-kernel-npu'", "ROOT=Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source')")
a=s.index('    if not l2:\n'); b=s.index('    q=case.inputs[0]',a)
s=s[:a]+'''    import importlib, inspect, sys
    sys.path.insert(0,str(ROOT/'python/sgl_kernel_npu'))
    module=importlib.import_module('sgl_kernel_npu.fla.chunk')
    fn=getattr(module,name)
    assert Path(module.__file__).resolve()==(ROOT/HOST).resolve()
    assert Path(inspect.unwrap(fn).__code__.co_filename).resolve()==(ROOT/HOST).resolve()
'''+s[b:]
(P/'integration700_sgl_gdn_remote38.py').write_text(s)
s=(old/'run_case.py').read_text().replace("W=Path('/root/dyq/workspace')","W=Path('/opt/tileops-all-pairs/workspace')").replace('sgl_gdn_recovery03','sgl_gdn_remote38')
s=s.replace('baseline,op,cid,output=sys.argv[1:]','op,cid,label=sys.argv[1:]; baseline="sgl-kernel-npu"; output=str(R/"runs"/(label+"--"+op+"--"+cid)/"output")')
s=s.replace("os.environ['ASCEND_CUSTOM_OPP_PATH']=str(W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math')",'')
s=s.replace("import importlib.util", "sys.path.extend(['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(W/'.cache/all-pairs/remote_native_tla/python-deps'),str(W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/recovery_1622/tasks/fla-bf16/runtime/deps')])\nimport time\n(work.parent/'acquired.json').write_text(json.dumps({'utc_epoch':time.time(),'metadata':m}))\nimport importlib.util")
s=s.replace("receipt.write_text(json.dumps(row,indent=2))", "receipt.write_text(json.dumps(row,indent=2))\n  (work.parent/'checker.json').write_text(json.dumps(row,indent=2))")
(P/'run_case.py').write_text(s)
(P/'probe01.json').write_text('[["DeltaNetFwdOp","decode-fresh-bfloat16"]]')
