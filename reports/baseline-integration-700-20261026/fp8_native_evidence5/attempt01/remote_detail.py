import pathlib,json,hashlib,re
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense'
files=['run_fp8_siblings.py','run_fp8_bmm.py','fp8_siblings01_guard.log','fp8_siblings01_guard.exit','ORIGINAL_DEVICE_BINARIES.json','run_private.py']
o={'files':{f:(P/f).read_text() for f in files},'logs':[]}
f=P/'fp8_siblings01/ASCEND_PROCESS_LOG_PATH/debug/plog/plog-2934886_20261026142154320.log'
ls=f.read_text(errors='replace').splitlines(); hits=[];window=[]
for i,l in enumerate(ls,1):
 m=re.search(r'2026-10-26-14:21:(\d\d)\.(\d{3})\.(\d{3})',l)
 if m and 'GroupedMatmul' in l and int(m[1]+m[2]+m[3])<=54419640: hits.append({'line':i,'text':l})
 if i<=10:window.append(l)
o['logs']=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'first_lines':window,'pre_last_finish_grouped_hits':hits}]
print(json.dumps(o))
