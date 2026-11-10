"""Current CPU metadata integrity checks; never substitute for device receipts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import PROVIDERS, native
from .metadata import inventory, parameters
from .plans import assess
from .omission_contracts import records, eligible

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    def passed(name,detail):checks.append({'check':name,'status':'pass','detail':detail})
    for path in HERE.glob('*.py'):
        compile(path.read_text(),str(path),'exec',dont_inherit=True)
    passed('syntax','All provider Python files compiled in memory without execution')
    code='''
import importlib.abc,sys,inspect
sys.path.insert(0,sys.argv[1])
class ForbidRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torch_npu','tilelang','acl','tvm'}:
            raise RuntimeError('Forbidden eager runtime import: '+fullname)
sys.meta_path.insert(0,ForbidRuntime())
from benchmarks.ascend_all_pairs.tla import PROVIDERS
for factory in PROVIDERS.values():
    assert list(inspect.signature(factory).parameters)==['case']
print(len(PROVIDERS))
'''
    result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,
                           str(native.WORKSPACE/'TileOPs')],capture_output=True,text=True,check=True)
    assert int(result.stdout)==len(PROVIDERS)
    passed('lazy_import',f'{len(PROVIDERS)} factories; isolated -I -S child denied runtime imports')
    inv=inventory();coverage=json.loads((HERE/'coverage.json').read_text())
    keys=lambda rows:{(r['op'],r['case_id']) for r in rows}
    assert keys(inv['cases'])==keys(coverage['cases'])
    assert len(coverage['cases'])==len(keys(coverage['cases']))==2062
    assert {r['op'] for r in inv['operators']}=={r['op'] for r in coverage['operators']}
    assert len(coverage['operators'])==205
    expected={}
    for row in inv['cases']:
        params,_=parameters(row)
        expected[(row['op'],row['case_id'])]=assess(row['op'],params).supported
    for row in coverage['cases']:
        assert row['case_adapter_available']==expected[(row['op'],row['case_id'])]
    supported={key for key,value in expected.items() if value}
    assert {op for op,_ in supported}==set(PROVIDERS)
    passed('coverage',f'205 operators, 2062 cases, {len(PROVIDERS)} providers, {len(supported)} supported exact cases')
    omission=records()
    assert len(keys(omission))==len(omission)==168
    for row in omission:
        assert eligible(row['op'],row['params'])
        assert (row['op'],row['case_id']) in supported
    passed('omission_contracts','168 distinct reviewed contracts are registered; no numerical claim')
    historical=json.loads((HERE/'static_case_receipts.json').read_text())['cases']
    static=json.loads((HERE/'static_validation.json').read_text())
    assert len(historical)==len(keys(historical))==static['factory_metadata_pass']==715
    assert static['run_metadata_pass']==715 and static['metadata_failures']==0
    assert keys(historical)<=supported
    assert not static['real_runtime_imports'] and not static['numerical_validation']
    passed('historical_receipts','715 original metadata receipts retained; their old adapter hashes are not current acceptance')
    catalogue=json.loads((HERE/'source_catalog.json').read_text())
    owned=[r for r in catalogue if '/3rdparty/' not in r['file']]
    for record in owned:
        assert hashlib.sha256((native.WORKSPACE/record['file']).read_bytes()).hexdigest()==record['sha256'],record['file']
    passed('vendor_preservation',f'{len(owned)} owned original files remain byte-identical')
    assert coverage['summary']['case_status_counts']==dict(Counter(r['status'] for r in coverage['cases']))
    assert hashlib.sha256((HERE/'inventory_snapshot.json').read_bytes()).hexdigest()==coverage['inventory_snapshot']['sha256']
    passed('counts_and_inventory_hash','Current coverage counts and immutable inventory snapshot agree')
    report={'schema':'tla-current-cpu-integrity/v2','generated_utc':datetime.now(timezone.utc).isoformat(),
            'npu_used':False,'numerical_validation':False,'result':'pass','checks':checks,
            'coverage_sha256':hashlib.sha256((HERE/'coverage.json').read_bytes()).hexdigest()}
    (HERE/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
