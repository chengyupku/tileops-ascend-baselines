import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from pathlib import Path
import sys,json,hashlib,datetime,inspect
W=Path(__file__).resolve().parent;R=W.parents[2];sys.path[:0]=[str(R/'TileOPs'),str(R/'TileOPs/src')]
import torch
assert '+cpu' in torch.__version__
torch.set_num_threads(4)
from benchmarks.api import cases
from workloads.numerics import compare_outputs,reference_tolerance
from workloads.device import set_run_device
set_run_device('cpu')
A=W/sys.argv[1];receipt=json.loads((A/'checker.json').read_text());case=next(c for c in cases('GDNFwdOp') if c.id==receipt['case_id'])
inputs=torch.load(A/'inputs.pt',map_location='cpu',weights_only=True);outputs=torch.load(A/'outputs.pt',map_location='cpu',weights_only=True);case.__dict__['inputs']=inputs
evidence=case.verification
# Local canonical source must match actual remote checker oracle before replay.
for rel,h in receipt['oracle_source_sha256'].items():
 assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel
expected=case.reference(*inputs);torch.save(expected,A/'cpu_expected.pt')
rows=[]
for i,(got,target) in enumerate(zip(outputs,expected,strict=True)):
 row=dict(index=i,name=['o','final_state'][i],shape=list(got.shape),dtype=str(got.dtype))
 probe=list(expected);probe[i]=got
 try:compare_outputs(tuple(probe),expected,evidence);row['passed']=True
 except AssertionError as e:row.update(passed=False,error=str(e))
 closure=inspect.getclosurevars(evidence.validator).nonlocals
 tol={'atol':closure['output_atol' if i==0 else 'state_atol'],'rtol':closure['rtol']};row['tolerance']=tol
 comparison_target=torch.nextafter(target,got) if i==0 and inputs[0].shape[1]==1 else target
 row['adjacent_rounding_allowance']=i==0 and inputs[0].shape[1]==1
 delta=(got.double()-target.double()).abs();bad=~torch.isclose(got.float(),comparison_target.float(),**tol,equal_nan=True);flat=bad.flatten().nonzero().flatten()
 row.update(mismatches=int(bad.sum()),elements=got.numel(),max_abs=float(delta.max()))
 if len(flat):
  idx=int(flat[0]);row['first_deviation']={'flat_index':idx,'coordinate':list(torch.unravel_index(torch.tensor(idx),got.shape)[j].item() for j in range(got.ndim)),'actual':float(got.flatten()[idx]),'expected':float(target.flatten()[idx]),'abs_error':float(delta.flatten()[idx])}
 rows.append(row)
try:compare_outputs(outputs,expected,evidence);complete=True
except AssertionError:complete=False
bound=torch.load(A/'bound_inputs.pt',map_location='cpu',weights_only=True)
q,k,v,g,beta,initial,*_=inputs
binding_checks=dict(q_equal=torch.equal(q,bound[0]),k_equal=torch.equal(k,bound[1]),v_equal=torch.equal(v,bound[2]),gate_broadcast_exact=torch.equal(g.float()[...,None].expand_as(bound[3]),bound[3]),beta_exact=torch.equal(beta.float(),bound[4]),initial_exact=initial is None if bound[7] is None else torch.equal(initial,bound[7]))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=case.id,input_seed=receipt.get('input_seed'),capture='Actual device inputs and both outputs; no CPU RNG replay',oracle_hashes_match=True,evidence=repr(evidence),complete_outputs_passed=complete,outputs=rows,binding_checks=binding_checks,torch_version=torch.__version__,torch_npu_imported=any(x.startswith('torch_npu') for x in sys.modules))
assert not result['torch_npu_imported']
(A/'CPU_ANALYSIS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
