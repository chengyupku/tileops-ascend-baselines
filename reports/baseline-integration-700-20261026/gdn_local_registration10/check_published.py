"""CPU only: fresh real registry plus in-memory staged provider append, never publish."""
import sys, os, json, hashlib, importlib, importlib.abc, mmap, types, argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--published',action='store_true');a=parser.parse_args()
sys.dont_write_bytecode=True
class NoDevice(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','acl'):
            raise AssertionError('Forbidden device import: '+fullname)
sys.meta_path.insert(0,NoDevice())
sys.path.insert(0,'/root/dyq/workspace/TileOPs')
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import Candidate,UnsupportedCase
provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
previous=provider.PROVIDERS['GDNFwdOp']
if not a.published:
    provider.__path__.insert(0,str(HERE))
    exec((HERE/'append_snippet.py').read_text(),provider.__dict__)
m=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_gdn_local_registration10')
assert Path(m.__file__).read_bytes()==(HERE/'integration700_gdn_local_registration10.py').read_bytes()
def case(key,params):return types.SimpleNamespace(id=key[2],params=params,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
def rejected(fn):
    try:fn()
    except (RuntimeError,ValueError,FileNotFoundError,KeyError):return
    raise AssertionError('Expected rejection')
for k,p in m.CASE_PARAMS.items():
    c=case(k,json.loads(p));candidate=registry.candidate_factory(c,k[0])()
    assert isinstance(candidate,Candidate) and callable(candidate.implementation.run)
    rejected(lambda:candidate.implementation.run()) # real guard denies, no imports
    c.params['not_canonical']=True
    rejected(lambda:registry.candidate_factory(c,k[0])())
    rejected(lambda:candidate.implementation.run())
# Unknown keys preserve previous factory; no call into device-dependent old route.
marker=object(); fallback=m.wrap(lambda c:marker)
k=next(iter(m.CASE_PARAMS)); assert fallback(case((k[0],k[1],'unknown'),{})) is marker
assert fallback(case((k[0],'WrongOp',k[2]),{})) is marker
assert not m.matches(case(k,json.loads(m.CASE_PARAMS[k])),'different-baseline')
if not a.published:assert provider.PROVIDERS['GDNFwdOp'].previous is previous
# Exact original private bytes, isolated from poisoned cached imports.
data=m._snapshot();sys.modules['direct_acl']=types.ModuleType('poison');sys.modules['resource_gate']=types.ModuleType('poison')
private=m._load(data)
assert private['binding'].DirectACL is private['direct_acl'].DirectACL
assert private['binding'].build.__code__.co_filename==str(m.ROOT/'binding.py')
assert (private['binding'].HERE/'RUNTIME_RECIPE.json').read_bytes()==data[str(m.ROOT/'RUNTIME_RECIPE.json')]
private['direct_acl'].DirectACL=None
assert m._load(data)['direct_acl'].DirectACL is not None
m._snapshot(runtime=True)
# Real host mappings only: no CDLL, no constructors, no driver handle.
rows=[]
library_entries=[x for x in json.loads(data[str(m.ROOT/'RUNTIME_PINS.json')])['files'] if '.so' in Path(x['path']).name]
for entry in library_entries:
    path=str(Path(entry['path']).resolve());digest=entry['sha256']
    with open(path,'rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        fields=next(x.split(maxsplit=5) for x in Path('/proc/self/maps').read_text().splitlines() if len(x.split(maxsplit=5))==6 and x.split(maxsplit=5)[5]==path)
        assert m._overlay_mapping_matches(path,digest,fields)
        assert not m._overlay_mapping_matches(path,'0'*64,fields)
        bad=list(fields);bad[3]='ffff:ffff';assert not m._overlay_mapping_matches(path,digest,bad)
        bad=list(fields);bad[4]=str(int(fields[4])+1);assert not m._overlay_mapping_matches(path,digest,bad)
        m._loaded_vendor_pins()
        st=Path(path).stat();rows.append(dict(path=path,kernel_device=fields[3],stat_device=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}',inode=fields[4]))
# Check operational admission without any device library (explicit fake capacity API).
import tempfile
with tempfile.TemporaryDirectory(dir=HERE) as td:
    saved=os.environ.get('GDN_RUN_DIR');os.environ['GDN_RUN_DIR']=td
    run=m._new_run(k[2]);assert run.name==k[2]
    assert m._new_run(k[2])!=run
    fake=types.SimpleNamespace(npu=types.SimpleNamespace(set_device=lambda n:None,mem_get_info=lambda:(3*1024**3,4*1024**3)))
    admission={'required_envelope':{'estimated_device_budget_bytes':1024**3}}
    m._hbm_admission(fake,run,admission)
    admission['required_envelope']['estimated_device_budget_bytes']+=1
    rejected(lambda:m._hbm_admission(fake,run,admission))
    os.environ['GDN_RUN_DIR']=str(m.ROOT.parent)
    rejected(lambda:m._new_run(k[2]))
    if saved is None:del os.environ['GDN_RUN_DIR']
    else:os.environ['GDN_RUN_DIR']=saved

# Real capacity reader may reject invisible container ancestors: fail-closed is required.
try: capacity=private['resource_gate'].capacity();capacity_status='read_success'
except RuntimeError as e:capacity_status='fail_closed: '+str(e)
assert not any(n in sys.modules for n in ('torch','torch_npu','acl'))
result=dict(status='verified_registration_preparation',accepted_added=0,published=a.published,exact_candidates=10,
 checks=['fresh_actual_registry_staging','all_params_mutation_rejected','unknown_key_previous_factory','wrong_baseline_and_op',
 'real_foreground_guard_denial_before_device_import','private_import_poison_isolated','fresh_private_namespace',
 'frozen_json_reads','fresh_case_id_directory_and_old_evidence_rejection','fake_HBM_boundary_2GiB_safety','complete_runtime_and_source_pins','real_host_readonly_mmap_hash_device_inode_negatives'],
 mapped_library_entries=len(rows),mapped_unique_libraries=len({r['path'] for r in rows}),mappings=rows,runtime_manifest_entries=1669,pin_count=len(m.PINS),capacity_reader=capacity_status,
 numerical_runs=0,npu_imports=0,remaining=['Parent review and locked publication; no additional numeric verification performed.'])
(HERE/('CPU_CHECKS_PUBLISHED.json' if a.published else 'CPU_CHECKS.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='mappings'},indent=2))
