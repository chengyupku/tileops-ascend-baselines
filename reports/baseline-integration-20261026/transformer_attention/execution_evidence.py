from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;rows=[]
for receipt in [*P.glob('runs/*/output/checker.json'),*P.glob('batches/*/*/checker.json')]:
 d=json.loads(receipt.read_text())
 if d.get('status')!='verified':continue
 out=receipt.parent;events=[]
 for log in out.glob('ASCEND_PROCESS_LOG_PATH/**/*.log'):
  for line in log.read_text(errors='replace').splitlines():
   if ('compile Ascend C operator FlashAttentionScore success' in line or ('FlashAttentionScore' in line and 'aclopCompileAndExecute' in line) or ('te_flashattentionscore_' in line and ('launch' in line.lower() or 'register' in line.lower()))):events.append({'log':str(log),'line':line})
 loaded={}
 for line in (out/'maps.txt').read_text().splitlines():
  path=line.split()[-1]
  if path.startswith('/') and any(s in path for s in ['libcust_','libacl_op_compiler','libascendcl']):
   f=Path(path)
   if f.is_file():loaded[path]=hashlib.sha256(f.read_bytes()).hexdigest()
 rows.append({'key':[d['library'],d['op'],d['case_id']],'checker':str(receipt),'verification':d['verification'],'inputs_preserved':d['inputs_preserved'],'candidate_returned':all(x.get('returned') for x in d['observed_calls']['candidate']),'loaded_library_sha256':loaded,'native_events':events})
(P/'EXECUTION_EVIDENCE.json').write_text(json.dumps(rows,indent=2)+'\n');print({'verified_receipts':len(rows),'native_event_counts':[len(r['native_events']) for r in rows]})
