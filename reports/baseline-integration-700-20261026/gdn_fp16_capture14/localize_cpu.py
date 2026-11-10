import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from pathlib import Path
import torch,json
W=Path(__file__).resolve().parent;torch.set_num_threads(4)
for name in ['attempt01_decode','attempt02_varlen']:
 a=W/name;x=torch.load(a/'inputs.pt',weights_only=True);got=torch.load(a/'outputs.pt',weights_only=True);exp=torch.load(a/'cpu_expected.pt',weights_only=True);bound=torch.load(a/'bound_inputs.pt',weights_only=True);r=json.loads((a/'CPU_ANALYSIS.json').read_text());tol=r['outputs'][0]['tolerance'];target=torch.nextafter(exp[0],got[0]) if x[0].shape[1]==1 else exp[0]
 bad=~torch.isclose(got[0].float(),target.float(),**tol,equal_nan=True);counts=bad.sum(dim=(0,2,3));ids=counts.nonzero().flatten().tolist()
 d=dict(case=r['case'],attention_failed_token_indices=ids,attention_failed_token_counts=[int(counts[i]) for i in ids],offsets=None if x[6] is None else x[6].tolist(),offsets_cpu_is_none=x[7] is None,bound_offsets=bound[8],no_root_cause_claim=True)
 (a/'LOCALIZATION.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
