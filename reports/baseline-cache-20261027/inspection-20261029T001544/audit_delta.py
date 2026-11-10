from pathlib import Path
import json,sys,hashlib,subprocess
host=sys.argv[1];r=Path('reports/baseline-cache-20261027');d=r/'inspection-20261029T001544';h=r/host
old=json.load(open(d/(host+'-AUDIT.json')));lines=(h/'CACHE.jsonl').read_text().splitlines();n=old['checked'];prefix='\n'.join(lines[:n])+'\n'
assert hashlib.sha256(prefix.encode()).hexdigest()==old['snapshot_prefix_sha256']
cache=h/'CACHE_DELTA_20261029T001544.jsonl';cache.open('x').write(('\n'.join(lines[n:])+'\n') if lines[n:] else '')
print('delta_rows',host,len(lines)-n,flush=True)
subprocess.run(['python3','tools/audit_baseline_cache.py','--cache',str(cache),'--fresh-hashes','--output',str(d/(host+'-AUDIT-delta.json'))],check=True)
