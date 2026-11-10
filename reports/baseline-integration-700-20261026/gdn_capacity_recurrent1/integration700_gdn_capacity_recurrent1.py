"""Exact GDN recurrent key awaiting canonical result and parent acceptance; unchanged original complete ACLNN binding."""
from pathlib import Path
import json, importlib.util, sys
PRIVATE=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/gdn_capacity_recurrent1')
def build(case):
    package=PRIVATE/'recurrent_package'
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    transport=load('_integration700_gdn_capacity_recurrent1_transport',package/'direct_acl.py')
    old=sys.modules.get('direct_acl');sys.modules['direct_acl']=transport
    try: binding=load('_integration700_gdn_capacity_recurrent1_binding',package/'binding.py')
    finally:
        if old is None:sys.modules.pop('direct_acl',None)
        else:sys.modules['direct_acl']=old
    return binding.build(case)
def wrap(previous):
    def dispatch(case):
        keys=json.loads((PRIVATE/'CANDIDATE_CASES.json').read_text())
        if case.id in keys:return build(case)
        if previous is None:raise KeyError(case.id)
        return previous(case)
    return dispatch
