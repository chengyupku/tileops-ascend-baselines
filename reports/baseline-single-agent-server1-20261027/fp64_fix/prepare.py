from pathlib import Path
import json,hashlib,difflib,ast
P=Path(__file__).resolve().parent;W=P.parents[2]
original=(W/'TileOPs/scripts/ascend_pair_check.py').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='3b9dd9dde43e5ff6c101496b714bd9d2710a6758264d4f851dd74da829e4bb9f'
review=json.loads((P.parent/'NUMERIC_ATTRIBUTION_REVIEW.json').read_text())
keys=[r['key'] for r in review['rows'] if r['group']=='reference_fp64']
(P/'SCOPE.json').write_text(json.dumps(keys,indent=2)+'\n')
helper='''
def verify_fp64_on_cpu(torch, case, observed_run, impl_args, reset, library, records):
    """Device dispatch only: original reference and full validator, no math changes."""
    from workloads.device import run_device, set_run_device
    from workloads.numerics import Request, verify
    def host(value):
        if isinstance(value, torch.Tensor):
            return value.to('cpu')
        if isinstance(value, tuple): return tuple(host(v) for v in value)
        if isinstance(value, list): return [host(v) for v in value]
        if isinstance(value, dict): return {k:host(v) for k,v in value.items()}
        return value
    def reference(*inputs):
        previous, threads = run_device(), torch.get_num_threads()
        try:
            set_run_device('cpu')
            torch.set_num_threads(min(threads, 4))
            return case.reference(*host(inputs))
        finally:
            set_run_device(previous)
            torch.set_num_threads(threads)
    def candidate(*inputs):
        return host(observed_run(*inputs))
    return verify(observe_call(torch, reference, records), case.inputs,
                  evidence=case.verification,
                  requests={library:Request(candidate,impl_args,reset,preserve_inputs=True)},
                  preserve_reference_inputs=True)

'''
s=original.replace('ROOT = Path(__file__).resolve().parents[1]',"ROOT = Path('/opt/tileops-all-pairs/workspace/TileOPs')")
s=s.replace('def tensor_metadata(torch, value):',helper+'\ndef tensor_metadata(torch, value):')
needle='            else:\n                reference = case.reference\n'
assert s.count(needle)==1
s=s.replace(needle,'''            elif [args.library,args.op,args.case_id] in json.loads(Path(__file__).with_name('SCOPE.json').read_text()):
                row['reference_device'] = 'cpu'
                row['validator_device'] = 'cpu'
                row['independent_cpu_reference'] = True
                row['reference_engine'] = 'unchanged original FP64 reference and full validator on CPU'
                row['reference_device_reason'] = 'User authorized 11-key FP64 environment repair; NPU rejects DT_DOUBLE'
                row['canonical_verifier_sha256'] = '3b9dd9dde43e5ff6c101496b714bd9d2710a6758264d4f851dd74da829e4bb9f'
                checks = verify_fp64_on_cpu(torch,case,observed_run,impl_args,impl.reset,args.library,row['observed_calls']['reference'])
'''+needle)
# Run-scoped checker: require exact authorized key even though the rest is unchanged.
s=s.replace('    require_guard()','    require_guard()\n    assert [args.library,args.op,args.case_id] in json.loads(Path(__file__).with_name("SCOPE.json").read_text())')
ast.parse(s);(P/'checker.py').write_text(s)
(P/'checker.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='canonical',tofile='scoped_cpu_dispatch')))
for src,dst in [('reports/baseline-integration-700-20261026/gdn_oracle_probe6/integration700_sgl_gdn_remote38.py','sgl_adapter.py')]:
 t=(W/src).read_text();(P/dst).write_text(t)
print('prepared',len(keys),'exact keys')
