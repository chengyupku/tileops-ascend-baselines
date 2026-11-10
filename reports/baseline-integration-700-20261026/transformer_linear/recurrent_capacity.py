from pathlib import Path
import json,sys
W=Path(__file__).resolve().parent;R=W.parents[2];O=R/'reports/baseline-integration-20261026/transformer_gdn';sys.path.insert(0,str(W/'chunk_package'));from resource_gate import capacity
k='q35-gva-decode-b512-bfloat16';est=json.loads((O/'CASE_RESOURCE_ESTIMATES.json').read_text())[k];c=capacity();c['required_envelope']=est;(W/'RECURRENT_CAPACITY.json').write_text(json.dumps(c,indent=2));print(c['effective_available_host_bytes'],est['minimum_effective_available_host_bytes'])
if c['effective_available_host_bytes']<est['minimum_effective_available_host_bytes']:
 b=json.loads((W/'BLOCKERS.json').read_text());b['GDNFwdOp/'+k]={'reason':'Current effective host/cgroup capacity below original full recurrent case envelope; no unchanged blocked device retry.','api':'aclnnRecurrentKda','evidence':'RECURRENT_CAPACITY.json'};(W/'BLOCKERS.json').write_text(json.dumps(b,indent=2))
