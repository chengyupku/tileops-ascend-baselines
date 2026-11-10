import sys,runpy,json
from pathlib import Path
P=Path(__file__).resolve().parent
items=json.load(open(P/sys.argv[1]))
for cid,op,lib in items:
 sys.argv=[str(P/'run_dense.py'),cid,op,lib]
 try:runpy.run_path(str(P/'run_dense.py'),run_name='__main__')
 except SystemExit:pass
 receipt=P/'runs'/cid/'checker.json'
 if not receipt.exists() or json.load(open(receipt)).get('status')!='verified':break
