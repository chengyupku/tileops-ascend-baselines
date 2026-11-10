import json,hashlib,re
from pathlib import Path
W=Path('/root/dyq/workspace'); P=W/'reports/baseline-integration-700-20261026/nn_dequant_registration1'; D=P.parent/'nn_dequant_diagnostic_cpu1'
r=json.loads((D/'RUNTIME_EVIDENCE.json').read_text()); checker=json.loads((D/'attempt01/checker.json').read_text())
pins={str(W/k):v for k,v in r['original_native_sources'].items()}
for row in r['loaded_vendor_libraries']: pins[row['path']]=row['sha256']
config=W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math/op_impl/ai_core/tbe/config/ascend950/aic-ascend950-ops-info.json'
pins[str(config)]=json.loads((D/'CPU_PREFLIGHT.json').read_text())['paths_sha256'][str(config)]
# Include only transitive project headers for Mul, not other operators or entire vendor.
root=W/'.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math/op_impl/ai_core/tbe/allpairs_math_math_impl/ascendc'
pending=[Path(k) for k in pins if str(root) in k and Path(k).suffix in ('.h','.cpp')]; visited=set(); external=set()
while pending:
 f=pending.pop()
 if f in visited:continue
 visited.add(f)
 for inc in re.findall(r'^\s*#\s*include\s*["<]([^">]+)',f.read_text(),re.M):
  matches=[q for q in (f.parent/inc,root/'common'/inc,root/inc) if q.is_file()]
  if matches:
   q=matches[0].resolve();key=str(q.relative_to(W));expected=checker['source_sha256'].get(key)
   if expected is None: raise RuntimeError('Header absent from verified receipt: '+key)
   pins[str(q)]=expected;pending.append(q)
  else:external.add(inc)
(P/'adapter.py').write_bytes((D/'attempt01/adapter.py').read_bytes());pins[str(P/'adapter.py')]='582491f0a11ee28efb2aadddfe38896feee4c181845a63bd14346e20c2313ea1'
for k,h in pins.items():assert hashlib.sha256(Path(k).read_bytes()).hexdigest()==h,k
(P/'PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
(P/'PIN_SCOPE.json').write_text(json.dumps({'count':len(pins),'external_toolchain_headers':sorted(external),'scope':'Original Mul sources, transitive deployed project headers, graph/specs, config, actual loaded vendor libraries and byte-identical adapter; CANN toolchain remains environment prerequisite.'},indent=2)+'\n')
print(len(pins),hashlib.sha256((P/'PINS.json').read_bytes()).hexdigest())
