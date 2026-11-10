import pathlib,subprocess,json
A=pathlib.Path(__file__).resolve().parent;S='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code=r'''
import sys,pathlib,json,importlib,types,hashlib
sys.dont_write_bytecode=True;sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','ctypes','acl'):raise RuntimeError('Forbidden '+fullname)
sys.meta_path.insert(0,Block());p=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');g=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn');cid='q35-397b-tp8-1024-7168-bfloat16';reg=g.PRIVATE/'REGISTERED.json';old=reg.read_text();oldread=pathlib.Path.read_text
# Preview proposed allowlist bytes only in memory; actual provider selection is unchanged.
def preview(self,*a,**kw):return json.dumps(sorted(set(json.loads(old))|{cid})) if self==reg else oldread(self,*a,**kw)
pathlib.Path.read_text=preview;g.build=lambda case:('exact_gdn',case.id)
case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GDNFwdOp')));result=p.PROVIDERS['GDNFwdOp'](case);assert result==('exact_gdn',cid);pathlib.Path.read_text=oldread;assert reg.read_text()==old
q=pathlib.Path('/opt/tileops-all-pairs/workspace/TileOPs/src/tileops/manifest/primitives.py');txt=q.read_text();lines=txt.splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('def causal_topk_indices'));end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('def ')),start+70)
print(json.dumps({'proposed_exact_route':result,'registry_module':p.__file__,'registry_sha256':hashlib.sha256(pathlib.Path(p.__file__).read_bytes()).hexdigest(),'factory_module':g.__file__,'factory_sha256':hashlib.sha256(pathlib.Path(g.__file__).read_bytes()).hexdigest(),'device_imports':[],'shared_writes':False,'preview_only':True,'generator':{'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'start_line':start+1,'text':'\n'.join(lines[start:end])}}))
'''
p=subprocess.run([S,'openlibing-DevEnv-055409','PYTHONDONTWRITEBYTECODE=1 python3 -S -'],input=code,text=True,capture_output=True,timeout=40);(A/'GDN_PATCH_CPU_PREVIEW.json').write_text(p.stdout);print(p.returncode,p.stderr);print(json.loads(p.stdout)['generator']['text'])
