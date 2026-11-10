import json,hashlib,time
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'guard_snapshot.json').read_text());pid=s['child']['pid']
paths=sorted(set(line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if '/' in line))
vendor='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense/gemm/private_provider'
files=[p for p in paths if vendor in p or 'grouped_matmul' in p.lower() or 'libcust_opapi' in p]
out={'at_epoch':time.time(),'pid':pid,'mapped_paths':paths,'native_mapped_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files if Path(p).is_file()}}
(P/'RUNTIME_MAPS.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'pid':pid,'native_mapped_sha256':out['native_mapped_sha256']}))
