import ast, difflib, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
W=Path('/root/dyq/workspace'); P=Path(__file__).resolve().parent

def dump(n,x): (P/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
dump('HEARTBEAT.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'creating private patches','npu_used':False})
if (P/'FEEDBACK.txt').exists(): print((P/'FEEDBACK.txt').read_text())
files=['TileOPs/workloads/gemm.py','TileOPs/scripts/ascend_pair_check.py','TileOPs/workloads/linear_attention/gdn.py','TileOPs/workloads/numerics.py','TileOPs/workloads/device.py']
dump('SOURCE_HASHES.json',{f:sha(W/f) for f in files})
def proposal(f,new,name):
 old=(W/f).read_text(); ast.parse(new)
 (P/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+f,tofile='b/'+f)))
 (P/(name+'.py')).write_text(new)
s=(W/files[0]).read_text(); anchor='    flags = torch.backends.cuda.matmul\n'
new='''    # CUDA reduction flags do not constrain CPU BF16 backend accumulation.
    # Widen exactly, accumulate in FP32, and preserve the original output dtype.
    if a.device.type == b.device.type == "cpu" and a.dtype == b.dtype == torch.bfloat16:
        with torch.autocast(device_type="cpu", enabled=False):
            return torch.matmul(a.float(), b.float()).to(a.dtype)

'''
assert s.count(anchor)==1
proposal(files[0],s.replace(anchor,new+anchor),'gemm_cpu_bf16')
s=(W/files[1]).read_text()
helper='''def verify_gdn_prefill_on_cpu(torch, case, observed_run, impl_args, reset, library, records):
    """Run the unchanged GDN reference and complete validator on host tensors.

    The candidate still receives its original device arguments and reset callback.
    verify retains device-input preservation checks, while both complete outputs
    are copied to CPU without dtype conversion for the original validator.
    """
    from workloads.device import run_device, set_run_device
    from workloads.numerics import Request, verify

    def host(value):
        if isinstance(value, torch.Tensor):
            return value.to("cpu")
        if isinstance(value, tuple):
            return tuple(host(x) for x in value)
        if isinstance(value, list):
            return [host(x) for x in value]
        if isinstance(value, dict):
            return {k: host(v) for k, v in value.items()}
        return value

    def reference(*inputs):
        previous_device, threads = run_device(), torch.get_num_threads()
        try:
            set_run_device("cpu")
            torch.set_num_threads(min(threads, 4))
            return case.reference(*host(inputs))
        finally:
            set_run_device(previous_device)
            torch.set_num_threads(threads)

    def candidate(*inputs):
        return host(observed_run(*inputs))

    return verify(
        observe_call(torch, reference, records), case.inputs,
        evidence=case.verification,
        requests={library: Request(candidate, impl_args, reset, preserve_inputs=True)},
        preserve_reference_inputs=True,
    )


'''
anchor='def tensor_metadata(torch, value):'
assert s.count(anchor)==1
s=s.replace(anchor,helper+anchor)
anchor='''            else:
                reference = case.reference
'''
branch='''            elif (args.library == "sgl-kernel-npu" and args.op == "GDNFwdOp"
                  and case.inputs[0].shape[1] > 1):
                row["reference_device"] = "cpu"
                row["validator_device"] = "cpu"
                row["independent_cpu_reference"] = True
                row["reference_engine"] = "unchanged GDN FP64 recurrence and complete validator on CPU"
                row["reference_device_reason"] = "NPU MatMul rejects DT_DOUBLE in original prefill reference"
                checks = verify_gdn_prefill_on_cpu(
                    torch, case, observed_run, impl_args, impl.reset, args.library,
                    row["observed_calls"]["reference"],
                )
'''
assert s.count(anchor)==1
proposal(files[1],s.replace(anchor,branch+anchor),'gdn_cpu_dispatch')
print('Created two private patches; shared sources untouched.')
