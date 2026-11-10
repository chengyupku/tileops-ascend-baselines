from pathlib import Path
import json,hashlib,shutil,difflib
W=Path('/root/dyq/workspace'); R=W/'reports/baseline-integration-20261026/register_local'; P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn'; N=R.parent/'nn_norm'; M=R.parent/'nn_misc_math/recovery01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changes=[]; reloc=[]
def write(p,text):
 old=p.read_text() if p.exists() else ''; backup=None
 if p.exists():
  backup=R/'backups'/p.relative_to(P);backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 changes.append({'path':str(p),'backup':str(backup) if backup else None,'before_sha256':sha(backup) if backup else None,'after_sha256':sha(p)})
 with (R/'integration.patch').open('a') as f:f.writelines(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(p.relative_to(W)) if backup else '/dev/null',tofile=str(p.relative_to(W))))
def copy(src,dst,replacements=()):
 text=src.read_text()
 for a,b in replacements:assert a in text;text=text.replace(a,b)
 write(dst,text);reloc.append({'source':str(src),'destination':str(dst),'source_sha256':sha(src),'destination_sha256':sha(dst),'changes':list(replacements)})
(R/'integration.patch').write_text('')
V=P/'verified_20261026';write(V/'__init__.py','"""Isolated original API runtime for the 16 reviewed exact keys."""\n')
for name in ['common.py','graph.py','catalog.py','truth.py','tensorlist.py']:
 copy(M/'providers/ops_nn'/name,V/name)
copy(N/'packet/frozen/normalization.py',V/'normalization.py')
copy(M/'providers/ops_nn/normalization.py',V/'normalization_misc.py')
copy(N/'packet/frozen/ascend_graph_call.py',V/'ascend_graph_call.py')
copy(N/'packet/frozen/ascend_nn_direct.py',V/'ascend_nn_direct.py',[('from benchmarks.ascend_graph_call import GraphCall','from .ascend_graph_call import GraphCall')])
copy(N/'packet/frozen/ascend_rms_cases.py',V/'ascend_rms_cases.py',[('from benchmarks.ascend_nn_direct import opsnn_direct_rms','from .ascend_nn_direct import opsnn_direct_rms')])
records=[]
for row in json.loads((R/'SCOPE.json').read_text()):
 _,op,cid=row['key']; base=R.parent/row['group'];review=json.loads((base/'PARENT_REVIEW.json').read_text()); rows=review.get('verified_rows',review.get('new_numerically_verified'))
 receipt=next(x for x in rows if x['key']==row['key']); checker=Path(receipt['checker']); guard=Path(receipt['guard']); c=json.loads(checker.read_text())
 assert c['status']=='verified' and json.loads(guard.read_text())['returncode']==0
 factory=('ascend_rms_cases:opsnn_rms_cases' if op=='RMSNormFwdOp' else ('normalization:normalization' if row['group']=='nn_norm' else 'normalization_misc:'+('count_nonzero' if op=='CountNonzeroFwdOp' else 'normalization')))
 vendor=json.loads((N/'packet/VENDORS.json').read_text()).get(op,str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'))
 records.append(dict(key=row['key'],factory=factory,vendor=vendor,checker=str(checker),checker_sha256=sha(checker),guard=str(guard),guard_sha256=sha(guard),review=str(base/'PARENT_REVIEW.json'),review_sha256=sha(base/'PARENT_REVIEW.json')))
write(P/'accepted_verified_20261026.json',json.dumps({'records':records,'runtime_files':{str(Path(x['destination']).relative_to(P)):x['destination_sha256'] for x in reloc}},indent=2)+'\n')
write(P/'verified_runtime.py','''"""Exact reviewed routes; device imports remain lazy and provider-local."""
import hashlib
import importlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'accepted_verified_20261026.json').read_text())
RECORDS = {tuple(r['key'][1:]): r for r in DATA['records']}


def validate(record):
    from benchmarks.ascend_baselines import UnsupportedCase
    pins = {HERE / p: h for p, h in DATA['runtime_files'].items()}
    for label in ('checker', 'guard', 'review'):
        pins[Path(record[label])] = record[label + '_sha256']
    for path, expected in pins.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedCase('Reviewed runtime/receipt changed; revalidation required: ' + str(path))
    vendor = Path(record['vendor']).resolve()
    if (Path(os.environ.get('OPS_NN_VENDOR', '')).resolve() != vendor or
            os.environ.get('ASCEND_CUSTOM_OPP_PATH') != str(vendor)):
        raise UnsupportedCase('Select the reviewed vendor before device imports with ops_nn/run_verified.py: ' + str(vendor))


def make(case):
    record = RECORDS[(case._call.signature.name, case.id)]
    validate(record)
    module, function = record['factory'].split(':')
    if module != 'ascend_rms_cases':
        # Same isolated per-process cache location as both verified launchers.
        from .verified_20261026 import graph
        graph.WORKSPACE = Path('/proc') / str(os.getpid()) / 'cwd'
    adapter = importlib.import_module('.verified_20261026.' + module, __package__)
    return getattr(adapter, function)(case)


def install(providers):
    for op in {key[0] for key in RECORDS}:
        previous = providers.get(op)
        def factory(case, _op=op, _previous=previous):
            if (case._call.signature.name, case.id) in RECORDS:
                return make(case)
            if _previous is not None:
                return _previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No accepted exact ops-nn case: ' + _op + '/' + case.id)
        factory.__name__ = 'reviewed_20261026_' + op
        providers[op] = factory
''')
write(P/'__init__.py',(P/'__init__.py').read_text()+"\n# Add reviewed exact keys without replacing historical acceptance or fallbacks.\nfrom .verified_runtime import install as _install_verified\n_install_verified(PROVIDERS)\n")
(R/'CHANGED_FILES.json').write_text(json.dumps(changes,indent=2)+'\n');(R/'RELOCATION.json').write_text(json.dumps(reloc,indent=2)+'\n');(R/'REGISTRATION_MAP.json').write_text(json.dumps(records,indent=2)+'\n')
