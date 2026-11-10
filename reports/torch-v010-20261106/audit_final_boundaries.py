"""Inspect actual task records, not profiler's process-wide name dictionary."""
import json,pathlib,sys,tarfile,tempfile,collections
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from tools.kernel_task_profile_full32 import parse_profile
rows=[]
for host in ('server1','server2'):
 for archive in ('final-representatives.tar.gz',):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp)
   with tarfile.open(P/host/archive) as t:t.extractall(root,filter='data')
   cases=[root] if archive=='rope-a1.tar.gz' else sorted(root.iterdir())
   for case in cases:
    m=json.load(open(case/'measurement.json'));base='/opt/torch-v010-20261106/preflight/'+('rope-a1' if case==root else case.name)
    names=collections.Counter()
    for r in m['timing']['retained_profiles']:
     d=case/pathlib.Path(r['profile_path']).relative_to(base)
     for task in parse_profile(d,1)['profile_kernel_tasks']:names[task['name']]+=1
    op=m['key'][1];forbidden=[]
    if op in ('MLADenseBwdOp','MLAVarlenBwdOp'):forbidden=[n for n in names if 'LogSumExp' in n or 'Logsumexp' in n]
    if op=='RoPEFwdOp':forbidden=[n for n in names if any(x in n for x in ('Pow','Sin','Cos','Arange','Reciprocal'))]
    if op=='GQADenseBwdOp':forbidden=[n for n in names if 'FlashAttentionScore' in n and 'Grad' not in n]
    if op=='EngramGateConvBwdOp':forbidden=[n for n in names if 'aclnnConvolution_' in n or 'aclnnSigmoid_' in n or 'aclnnSilu_' in n]
    rows.append({'host':host,'key':m['key'],'task_names':dict(names),'unexpected_preparation_tasks':forbidden,'automated_exclusion_scope':op in ('RoPEFwdOp','GQADenseBwdOp','EngramGateConvBwdOp')})
(P/'PREFLIGHT_FINAL_BOUNDARY_TRACE_AUDIT.json').write_text(json.dumps(rows,indent=2))
assert not any(r['unexpected_preparation_tasks'] for r in rows),[(r['key'],r['unexpected_preparation_tasks']) for r in rows if r['unexpected_preparation_tasks']]
print('Actual trace boundary exclusions passed; other routes retain task inventories for source-level review')
