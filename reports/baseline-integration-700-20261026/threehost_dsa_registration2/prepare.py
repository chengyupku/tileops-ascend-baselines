from pathlib import Path
import json,hashlib,shutil,ast
O=Path(__file__).resolve().parent; E=O.parent/'threehost_attention_moe53/remote_evidence';F=E/'frozen/TileOPs';W='/opt/tileops-all-pairs/workspace';R=W+'/reports/baseline-integration-700-20261026/threehost_attention_moe53'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
v=json.loads((F/'benchmarks/ascend_all_pairs/ops_transformer/validated_manifest.json').read_text());save('FROZEN.json',v)
sources={}
for p in (O/'closure').rglob('*'):
 if p.is_file():sources['/'+str(p.relative_to(O/'closure'))]={'file':str(p.relative_to(O)),'sha256':sha(p)}
for rel in ['benchmarks/ascend_baselines.py','benchmarks/api.py','benchmarks/ascend_math_more.py','workloads/attention/dsa.py','workloads/numerics.py','scripts/ascend_pair_check.py']:
 p=F/rel;d=O/'closure'/R.lstrip('/')/'frozen/TileOPs'/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);sources[R+'/frozen/TileOPs/'+rel]={'file':str(d.relative_to(O)),'sha256':sha(d)}
for n in ['dsa.py','dsa.original.py']:
 d=O/'closure'/R.lstrip('/')/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(E/n,d);sources[R+'/'+n]={'file':str(d.relative_to(O)),'sha256':sha(d)}
# CPU manifest instantiation is pinned independently of runtime imports.
for p in (F/'src/tileops/manifest').rglob('*'):
 if p.is_file():
  d=O/'manifest_closure'/p.relative_to(F/'src');d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for rel in ['__init__.py']:
 d=O/'manifest_closure/tileops'/rel;shutil.copyfile(F/'src/tileops'/rel,d)
runtime=dict(v['groups']['attention']['pins']);runtime['/opt/tileops-all-pairs/control/private_python.py']=sources['/opt/tileops-all-pairs/control/private_python.py']['sha256']
for case in ['mla-d576-h128-kv96k-bfloat16-attempt06','mla-d576-h128-kv128k-bfloat16-attempt01']:
 n=json.loads((E/case/'native.json').read_text())
 for p,item in n['mapped_files'].items():
  if p in runtime:assert runtime[p]==item['sha256']
  runtime[p]=item['sha256']
save('PINS.json',dict(sources=sources,runtime=runtime,host={'machine':'aarch64','node':'897b29c557a5'},environment=v['groups']['attention']['environment']))
# Reuse reviewed guard, full-contract and lifetime machinery; customize only DSA loader/environment.
s=(O.parent/'threehost_native9_registration/registration_parent.py').read_text()
s=s.replace('threehost_native9_registration','threehost_dsa_registration2').replace('NATIVE9','DSA2').replace('native9','dsa2').replace('Only tested server1 ARM host','Only tested server2 ARM host')
a=s.index('FP8 =');b=s.index('\n\ndef checked',a)
s=s[:a]+"ORIGINAL = REPORT + '/threehost_attention_moe53'\nBASE = ORIGINAL + '/frozen/TileOPs/benchmarks/ascend_baselines.py'\nAPI = ORIGINAL + '/frozen/TileOPs/benchmarks/api.py'\nDSA = ORIGINAL + '/dsa.py'\nDATA_HASHES = {}\n"+s[b:]
a=s.index('def private_load(');b=s.index('\ndef _proc(',a)
s=s[:a]+'''def private_load(data):
    """Execute unchanged source bytes; private import wiring and hash metadata only."""
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    cfg=metadata()['FROZEN.json']; private={}; original_import=builtins.__import__
    names={'benchmarks.ascend_baselines':BASE,'benchmarks.api':API}
    names.update({'benchmarks.'+n:p for n,p in cfg['sources'].items() if n.startswith('ascend_')})
    def load(path):
        if path in private:return private[path]
        if path not in data:raise RuntimeError('Unpinned source: '+path)
        mod=types.ModuleType('_dsa2_'+hashlib.sha256(path.encode()).hexdigest()[:16])
        mod.__file__=str(DATA_ROOT/metadata()['PINS.json']['sources'][path]['file'])
        mod.__dict__['__builtins__']=dict(vars(builtins),__import__=private_import)
        private[path]=mod
        if path==API:mod.Implementation=type(_implementation(lambda:None,data))
        else:exec(compile(data[path],mod.__file__,'exec'),mod.__dict__)
        if path==BASE:mod.Candidate=Candidate;mod.UnsupportedCase=UnsupportedCase
        return mod
    def verified(group):
        if group!='attention':raise RuntimeError('Unsupported source group')
        # admit() verified the entire runtime pin set once for this candidate.
    bridge=types.SimpleNamespace(verify_pins=verified,load=lambda name:load(cfg['sources'][name]))
    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level==0 and name=='benchmarks.ascend_all_pairs.ops_transformer.validated_runtime':return bridge
        if level==0 and name in names:return load(names[name])
        return original_import(name,globals,locals,fromlist,level)
    return load

''' + s[b:]
s=s.replace("mod=load(FP8 if key(frozen)[0]=='BmmFP8FwdOp' else BF16)","mod=load(DSA)").replace("(mod.fp8_bmm if key(frozen)[0]=='BmmFP8FwdOp' else mod.make_candidate)(frozen)","mod.factory(frozen)")
s=s.replace("'ops-transformer::aclnnGroupedMatmulV5'","'ops-transformer::FlashAttentionScore'")
s=s.replace("            loaded_libraries()\n            return result","            import torch\n            torch.npu.synchronize()\n            loaded_libraries()\n            return result")
# Short proc alias prevents AF_UNIX path overflow while keeping temp physically inside isolated run.
s=s.replace("if Path(os.environ.get(name,'/invalid')).resolve()!=root/name:","if Path(os.environ.get(name,'/invalid')).resolve()!=root/name:")
# Explicit compiler and overlay import precedence checks before native initialization.
pos=s.index("    return tuple(sorted(os.environ.items())),str(root)")
s=s[:pos]+'''    import shutil
    cfg=metadata()['FROZEN.json']['groups']['attention']
    if os.environ.get('LD_LIBRARY_PATH')!=cfg['ld_library_path_prefix']:
        raise RuntimeError('Original server2 LD_LIBRARY_PATH required')
    if os.environ.get('ASCEND_CANN_PACKAGE_PATH')!='/usr/local/Ascend/cann-9.1.0':
        raise RuntimeError('Compiler SDK path required')
    for exe,expected in [('bisheng','/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng'),('ccec','/usr/local/Ascend/cann-9.1.0/compiler/ccec_compiler/bin/ccec')]:
        got=shutil.which(exe)
        if got is None or Path(got).resolve()!=Path(expected).resolve():raise RuntimeError('Compiler PATH mismatch: '+exe)
    overlay=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
    for package in ('torch','torch_npu'):
        candidates=[Path(p)/package/'__init__.py' for p in sys.path if (Path(p)/package/'__init__.py').is_file()]
        if not candidates or candidates[0].resolve()!=(overlay/package/'__init__.py').resolve():
            raise RuntimeError('Original overlay must precede base site-packages: '+package)
    if len(os.environ.get('TMPDIR',''))>60:raise RuntimeError('Use short /proc/self/cwd/TMPDIR alias')
''' + s[pos:]
(O/'registration.py').write_text(s)
print('prepared',len(sources),'source files;',len(runtime),'runtime pins')
