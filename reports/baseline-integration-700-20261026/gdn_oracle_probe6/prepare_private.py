from pathlib import Path
import hashlib,json,subprocess,shutil,difflib,ast
P=Path(__file__).resolve().parent; W=P.parents[2]; F=P.parent/'sgl_gdn_register30/frozen_verified'; Q=P.parent/'oracle_repair_proposal55'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((Q/'SOURCE_HASHES.json').read_text())
assert all(h(W/k)==v for k,v in expected.items())
original=(W/'TileOPs/scripts/ascend_pair_check.py').read_text()
(P/'checker.original.py').write_text(original)
shutil.copy2(Q/'gdn_cpu_dispatch.patch',P/'gdn_cpu_dispatch.patch')
shutil.copy2(P/'checker.original.py',P/'checker.private.py')
r=subprocess.run(['patch','--batch',str(P/'checker.private.py'),str(P/'gdn_cpu_dispatch.patch')],capture_output=True,text=True)
(P/'PATCH_APPLY.json').write_text(json.dumps(dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)); assert r.returncode==0
s=(P/'checker.private.py').read_text(); (P/'checker.dispatch_only.py').write_text(s)
s=s.replace('ROOT = Path(__file__).resolve().parents[1]',"ROOT = Path('/opt/tileops-all-pairs/workspace/TileOPs')")
s=s.replace('        result = run(*args, **kwargs)', '''        from capture import capture
        label = 'reference' if run.__name__ == 'reference' else 'candidate'
        capture(torch, label + '_inputs_before', args)
        try:
            result = run(*args, **kwargs)
        finally:
            capture(torch, label + '_inputs_after', args)
        capture(torch, label + '_outputs', result)''')
s=s.replace('            return case.reference(*host(inputs))','''            from capture import capture
            cpu_inputs = host(inputs)
            capture(torch, 'reference_cpu_inputs_before', cpu_inputs)
            try:
                return case.reference(*cpu_inputs)
            finally:
                capture(torch, 'reference_cpu_inputs_after', cpu_inputs)''')
s=s.replace('row["status"] = "verified"','row["status"] = "private_checker_diagnostic_pass"')
(P/'checker.private.py').write_text(s)
(P/'checker.private.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original/ascend_pair_check.py',tofile='private/checker.private.py')))
for name in ['integration700_sgl_gdn_remote38.py','route_install.py','native_log_capture.py','baseline-run']:
 shutil.copy2(F/name,P/name)
s=(F/'run_case.original_numerics.frozen.py').read_text()
s=s.replace("os.environ.pop('ASCEND_CUSTOM_OPP_PATH',None)","os.environ.pop('ASCEND_CUSTOM_OPP_PATH',None)\nimport torch\ntorch.set_num_threads(4)\ntorch.set_num_interop_threads(4)")
s=s.replace("from native_log_capture import NativeLogCapture",'''from native_log_capture import NativeLogCapture
from capture import install_native_trace
install_native_trace(work)
''')
s=s.replace("(work/'maps.txt').write_text(Path('/proc/self/maps').read_text())", "(work/'maps.txt').write_text(Path('/proc/self/maps').read_text())\n sys.setprofile(None)")
(P/'run_case.py').write_text(s)
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gdn_oracle_probe6'
(P/'CHECKER_PIN.json').write_text(json.dumps(dict(path=remote+'/checker.private.py',sha256=h(P/'checker.private.py')),indent=2))
for p in P.glob('*.py'):ast.parse(p.read_text())
(P/'PREFLIGHT_LOCAL.json').write_text(json.dumps(dict(original_hashes=expected,private_checker_sha256=h(P/'checker.private.py'),frozen_adapter_sha256=h(P/'integration700_sgl_gdn_remote38.py'),patch_sha256=h(P/'gdn_cpu_dispatch.patch'),cpu_preflight='stdlib only; torch/torch_npu/acl never imported',parameters=dict(B=1,S=4096,H=16,HV=16,K=64,V=64,dtype='bfloat16',initial_state=None),input_bytes=3*1*4096*16*64*2+2*4096*16*2,output_bytes=4096*16*64*2+16*64*64*4,reference_chunk_fp64_tensor_bytes=16*64*64*8,estimated_cpu_working_set='under 512 MiB tensor storage plus runtime/libraries; capture copies transient; 64 unchanged FP64 chunks',scope_count=6,planned_probes=1,accepted_added=0),indent=2))
