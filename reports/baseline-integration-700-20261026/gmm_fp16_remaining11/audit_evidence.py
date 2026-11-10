from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent;out=[]
for q in P.iterdir():
 if not q.is_dir() or not (q/'representative.json').exists():continue
 r=json.loads((q/'representative.json').read_text());a={'attempt':q.name,'status':r['status'],'checker_sha256':hashlib.sha256((q/'representative.json').read_bytes()).hexdigest()}
 if (q/'EXECUTION.json').exists():
  e=json.loads((q/'EXECUTION.json').read_text());a.update(checker_exit=e['checker_exit_code'],frozen_unchanged=e['frozen_unchanged'],loaded_private_maps=sorted(set(f for f in e['loaded_files'] if '/transformer_dense/gemm/private_provider/' in f)),capture_counts=e['capture_counts'])
 if (q/'CAPTURE.json').exists():
  c=json.loads((q/'CAPTURE.json').read_text());matches={}
  for n,m in c.items():
   if n.startswith('reference_0_input'):
    other='candidate_0_input'+n.split('input')[1]
    if other in c:matches[n]=m.get('sha256')==c[other].get('sha256') and m.get('shape')==c[other].get('shape')
  a['actual_reference_candidate_inputs_identical']=matches
 if (q/'native_dispatch_excerpt.log').exists() and (q/'FROZEN.json').exists():
  text=(q/'native_dispatch_excerpt.log').read_text();f=json.loads((q/'FROZEN.json').read_text())['sha256'];bins=[]
  for line in text.splitlines():
   if 'BinLoadImpl' in line and 'Register Kernel succ' in line:
    paths=re.findall(r'(/\S+\.o)',line)
    for path in paths:
     if 'GroupedMatmul' in path:bins.append({'path':path,'sha256':f.get(path),'driver_line':line})
  a['actual_native_bins']=bins
 if r['status']=='verified':
  assert r['inputs_preserved'] and r['verification']['checked_outputs']==r['verification']['total_outputs']
  assert not r['provider_source_changed'] and not r['oracle_source_changed']
  if 'frozen_unchanged' in a:assert a['frozen_unchanged']
  if 'actual_native_bins' in a:assert a['actual_native_bins'] and all(b['sha256'] for b in a['actual_native_bins'])
 out.append(a)
(P/'CHECK_AUDIT.json').write_text(json.dumps(out,indent=2));print('audited',len(out),'receipts')
