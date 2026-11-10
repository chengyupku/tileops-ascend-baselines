"""Build exact-task preaccepted ledger from immutable raw; never operates remote hosts."""
import pathlib,json,tarfile,tempfile,hashlib,sys,time,statistics,re
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from tools.kernel_task_profile_full32 import parse_profile,trace_digest,PARSER_REVISION
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
queue={r['task']['id']:r for r in json.load(open(P/'QUEUE.json'))['tasks']}
require_final={'GQADenseBwdOp','GLAChunkBwdOp','EngramGateConvBwdOp','DeltaNetChunkBwdOp','MLADenseBwdOp','MLAVarlenBwdOp'}
selected={};superseded=[]
for host in ('server1','server2'):
 for archive_name,sha_name in [('rope-a1','rope-archive-sha'),('representatives','representatives-sha'),('expanded-representatives','expanded-representatives-sha'),('final-representatives','final-representatives-sha'),('gelu-representatives','gelu-representatives-sha')]:
  archive=P/host/(archive_name+'.tar.gz')
  if not archive.exists():
   assert archive_name=='gelu-representatives' and host=='server1',str(archive)
   continue
  assert sha(archive)==(P/host/(sha_name+'.txt')).read_text().split()[0]
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp)
   with tarfile.open(archive) as t:t.extractall(root,filter='data')
   if archive_name=='rope-a1':
    superseded.append({'host':host,'archive':str(archive),'reason':'Preflight stdout/stderr not in archive; retain validation evidence but remeasure formal task with full log'})
    continue
   cases=sorted(root.iterdir())
   for case in cases:
    m=json.load(open(case/'measurement.json'));tid=m['task_id'];op=m['key'][1]
    if op in require_final and '-v3-' not in case.name:
     superseded.append({'host':host,'task_id':tid,'attempt':case.name,'reason':'final execution boundary required'});continue
    q=queue[tid];task=json.load(open(case/'task.json')) if (case/'task.json').exists() else json.load(open(P/'PREFLIGHT_TASK.json'))
    assert task['id']==tid and task['key']==m['key']==q['task']['key'] and task['case']==q['task']['case']
    g=json.load(open(case/'guard.json'));c=json.load(open(case/'correctness.json'))
    assert g['performance_valid'] and g['phase']=='finished' and g['exit_code']==g['workload_exit_code']==0
    assert g['owner'].startswith('torch-v010/'+host+'/')
    assert c['status']=='verified' and c['inputs_preserved'] and m['inputs_preserved'] and sha(case/'correctness.json')==m['correctness_sha256']
    assert (m['hardware']['cube'],m['hardware']['vector'])==(32,64)
    assert m['source_sha256'] in {sha(f) for f in P.glob('SOURCES.v*.json')}
    log=(case/'device.log').read_text(errors='replace') if (case/'device.log').exists() else ''
    assert 'fall back to run on the CPU' not in log and 'npu_cpu_fallback' not in log
    profiles=m['timing']['retained_profiles'];samples=m['timing']['latency_samples_ms'];assert len(profiles)==len(samples)==15
    base='/opt/torch-v010-20261106/preflight/'+('rope-a1' if case==root else case.name)
    for r,v in zip(profiles,samples):
     d=case/pathlib.Path(r['profile_path']).relative_to(base)
     assert r['parser_revision']==PARSER_REVISION and trace_digest(d)==r['profile_trace_sha256']
     assert abs(parse_profile(d,1)['latency_ms']-v)<1e-9
    assert abs(statistics.mean(samples)-m['timing']['latency_ms'])<1e-9
    selected[tid]={'accepted':True,'verified':True,'preflight_audit':'PREFLIGHT_IMPORT_AUDIT.json','host':host,'archive':str(archive),'sha256':sha(archive),'archive_member':'.' if case==root else case.name,'source_sha256':m['source_sha256'],'key':m['key'],'at':time.time(),'measurement':{'latency_ms':m['timing']['latency_ms'],'key':m['key'],'raw_windows':15},'guard_owner':g['owner'],'reused_without_rerun':True}
assert len(selected)==38,len(selected)
for host in ('server1','server2'):
 (P/host/'PREACCEPTED.json').write_text(json.dumps({k:v for k,v in selected.items() if v['host']==host},indent=2))
(P/'PREFLIGHT_IMPORT_AUDIT.json').write_text(json.dumps({'accepted_unique_tasks':len(selected),'total_queue':len(queue),'superseded_attempts':superseded,'checks':['exact current key and parameters','correctness receipt hash','32AIC/64AIV','valid finished guard','frozen source manifest','archive SHA','all15 raw SHA and frozen v3 replay','CPU fallback rejected where device log retained'],'limitation':'input preservation retains campaign sampled-byte policy; not exhaustive input mutation audit','selected':selected},indent=2))
print('preaccepted',len(selected))
