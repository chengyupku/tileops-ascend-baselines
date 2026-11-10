import pathlib,json,subprocess
A=pathlib.Path(__file__).resolve().parent;idx=json.loads((A/'REMOTE_INDEX.json').read_text());S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code=r'''
import pathlib,json,hashlib,difflib
W=pathlib.Path('/opt/tileops-all-pairs/workspace');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();out={'native':[],'files':{}}
for l in LOGS:
 p=pathlib.Path(l['path']);lines=p.read_text(errors='replace').splitlines();groups={}
 for label,pred in [('vendor',lambda s:'classicpilot_attention_transformer' in s and ('flash_attention' in s.lower() or 'libop' in s)),('launch',lambda s:('launch' in s.lower() or 'execute' in s.lower()) and ('flashattention' in s.lower() or 'flash_attention' in s.lower())),('binary',lambda s:'te_flashattentionscore' in s.lower() and ('.o' in s or 'kernel' in s.lower()))]:
  hits=[{'line':i+1,'text':s} for i,s in enumerate(lines) if pred(s)];groups[label]=hits[-4:]
 out['native'].append({'case_id':l['case_id'],'path':str(p),'sha256':sha(p),'evidence':groups})
for rel in ['tools/baseline_protocol.py','TileOPs/src/tileops/manifest/spec/attention.yaml','reports/baseline-integration-700-20261026/transformer_linear/integration700_transformer_linear_gdn.py','reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp8-1024-7168-bfloat16/checker.json','reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp8-1024-7168-bfloat16/GUARD_RECEIPT.json']:
 p=W/rel;out['files'][rel]={'sha256':sha(p),'text':p.read_text()}
c=json.loads(out['files']['reports/baseline-integration-700-20261026/transformer_linear/q35-397b-tp8-1024-7168-bfloat16/checker.json']['text']);before=c['provider_source_sha256'];after=c['provider_source_sha256_after'];out['gdn_changes']={p:{'before':before.get(p),'after':after.get(p)} for p in before.keys()|after.keys() if before.get(p)!=after.get(p)}
# Find only provider snapshots in original team dirs; no history/memory scan.
wanted={h for v in out['gdn_changes'].values() for h in v.values() if h};found={}
for team in ['transformer_attention_misc','transformer_linear']:
 for p in (W/'reports/baseline-integration-700-20261026'/team).glob('*.py'):
  h=sha(p)
  if h in wanted:found[h]={'path':str(p),'text':p.read_text()}
out['gdn_snapshots']=found
out['gdn_critical']=[]
for p,h in c['source_sha256'].items():
 if any(s in p for s in ['chunk_package/binding.py','chunk_package/entry.py','integration700_transformer_linear_gdn.py']):
  q=pathlib.Path(p);q=q if q.is_absolute() else W/q;out['gdn_critical'].append({'path':str(q),'expected':h,'sha256':sha(q),'text':q.read_text()})
print(json.dumps(out))
'''.replace('LOGS',repr(idx['native_logs']))
p=subprocess.run([S,'openlibing-DevEnv-055409','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=90);(A/'EXTRA.json').write_text(p.stdout);print('exit',p.returncode,p.stderr)
