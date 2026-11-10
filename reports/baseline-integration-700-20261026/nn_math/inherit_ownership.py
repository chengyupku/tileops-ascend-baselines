"""Recover same-process native provenance from persistent CANN log directory."""
import datetime,json,pathlib,re,sys
P=pathlib.Path(__file__).resolve().parent;first=sys.argv[1];attempts=sys.argv[2:];logs=list((P/first/'ASCEND_PROCESS_LOG_PATH').rglob('*.log'))
compile_rows=[]
for f in logs:
 for i,l in enumerate(f.open(errors='replace'),1):
  if 'single op compile success' in l and any(x in l for x in ['allpairs_nn_impl','classicpilot_nn_impl']):compile_rows.append(dict(file=str(f),line=i,text=l.strip()))
for l in (P/first/'native.jsonl').read_text().splitlines():
 if 'single op compile success' in l:compile_rows.append(json.loads(l))
assert compile_rows
for a in attempts:
 c=json.loads((P/a/'checker.json').read_text());lo=datetime.datetime.fromisoformat(c['started_at']).replace(tzinfo=None);hi=datetime.datetime.fromisoformat(c['finished_at']).replace(tzinfo=None);hits=[]
 for f in logs:
  for i,l in enumerate(f.open(errors='replace'),1):
   if 'kernel_name=te_' not in l:continue
   m=re.search(r'(2026-\d\d-\d\d-\d\d:\d\d:\d\d\.\d{3}\.\d{3})',l)
   if m:
    t=datetime.datetime.strptime(m[1][:23]+m[1][24:],'%Y-%m-%d-%H:%M:%S.%f')
    if lo<=t<=hi:hits.append(dict(file=str(f),line=i,text=l.strip()))
 assert hits,a
 (P/a/'OWNERSHIP.json').write_text(json.dumps(dict(reason='Same-process original graph JIT; persistent initial CANN logging directory. Launch timestamps within exact checker interval.',shared_compile=compile_rows,matching_launches=len(hits),launch_evidence=hits[:2]+hits[-2:]),indent=2));print(a,len(hits))
